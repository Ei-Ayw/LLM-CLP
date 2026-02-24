import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# =============================================================================
# ### 核心训练脚本：train_deberta_mtl_stage2.py (AMP加速版) ###
# 设计说明：
# 执行身份感知重加权微调。集成了 FP16 加速。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model_deberta_v3_mtl import DebertaV3MTL
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping

def weighted_toxicity_loss(logits, targets, has_id, w_identity, w_id_toxic):
    weights = torch.ones_like(targets)
    has_id_mask = has_id.unsqueeze(-1).bool()
    is_toxic = targets >= 0.5
    weights[(~is_toxic) & has_id_mask] = w_identity
    weights[is_toxic & has_id_mask] = w_id_toxic
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(logits, targets)
    return (loss * weights).mean()

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps, alpha, beta, w_identity, w_id_toxic, no_reweight=False):
    model.train()
    criterion_aux = nn.BCEWithLogitsLoss()
    total_loss = 0
    optimizer.zero_grad() # [Fix] 显式初始化梯度
    pbar = tqdm(loader, desc="[Train S2 AMP]")
    for i, batch in enumerate(pbar):
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        y_tox, y_sub, y_id, has_id = batch['y_tox'].to(device).unsqueeze(-1), batch['y_sub'].to(device), batch['y_id'].to(device), batch['has_id'].to(device)
        
        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            if no_reweight:
                l_tox = criterion_aux(out['logits_tox'], y_tox)
            else:
                l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, w_identity, w_id_toxic)
            l_sub, l_id = criterion_aux(out['logits_sub'], y_sub), criterion_aux(out['logits_id'], y_id)
            loss = l_tox + alpha * l_sub + beta * l_id
        
        loss = loss / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
    return total_loss / len(loader)

def evaluate(model, loader, device, w_identity=2.5, w_id_toxic=1.5, no_reweight=False):
    """
    [Fix] 评估函数与训练保持一致的损失函数
    - 修复前：训练用 weighted_toxicity_loss，评估用 BCE，导致 checkpoint 选择偏差
    - 修复后：评估也使用 weighted_toxicity_loss，确保 Early Stopping 基于正确的指标
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval S2]"):
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            has_id = batch['has_id'].to(device)
            out = model(ids, mask)
            if no_reweight:
                loss = nn.BCEWithLogitsLoss()(out['logits_tox'], y_tox)
            else:
                loss = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, w_identity, w_id_toxic)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="DeBERTa-V3 Stage 2 MTL (AMP)")
    parser.add_argument("--s1_checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.1, help="MTL Weight for Subtypes [Fixed: was 0.5, now 0.1 to match S1]")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--grad_accum", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--w_identity", type=float, default=2.5)
    parser.add_argument("--w_id_toxic", type=float, default=1.5)
    parser.add_argument("--no_reweight", action="store_true")
    parser.add_argument("--no_bar", action="store_true")

    # Ablation Flags
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--ablation_tag", type=str, default=None)

    args = parser.parse_args()
    
    # --- DDP Initialization ---
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        is_main_process = (args.local_rank == 0)
    else:
        # Fallback for non-DDP run (though we expect torchrun)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
    
    # 静默模式处理 (仅主进程显示)
    if args.no_bar or (not is_main_process):
        global tqdm
        from tqdm import tqdm as _tqdm
        tqdm = lambda *args, **kwargs: _tqdm(*args, **kwargs, disable=True)

    # [Performance] Hardware Acceleration - Safe Mode
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(args.seed)
    
    # Suffix Construction
    suffix = ('_NoReweight' if args.no_reweight else '')
    if args.ablation_tag: suffix += f"_{args.ablation_tag}"
    
    save_name = f"DebertaV3MTL_S2{suffix}_Sample{args.sample_size}_{datetime.now().strftime('%m%d_%H%M')}"
    save_path = get_model_path(save_name + ".pth")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len, augment=not args.no_aug) # 开启简单数据增强
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len, augment=False)
    # --- DDP Data Loading ---
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=train_sampler,
        num_workers=0, 
        pin_memory=True
    )
    # S2 的验证集也只在主进程跑简单点
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    model = DebertaV3MTL(args.model_name).to(device)
    if os.path.exists(args.s1_checkpoint):
        model.load_state_dict(torch.load(args.s1_checkpoint, map_location=device))
        print(f"  [Success] 加载 S1 权重成功。")

    # --- DDP Model Wrapping ---
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
    else:
        world_size = dist.get_world_size()
        num_steps = int(len(train_ds) / args.batch_size / world_size * args.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    early_stopping, best_val_loss = EarlyStopping(patience=args.early_patience), float('inf')
    loss_history = {"train": [], "val": []}
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if is_main_process: print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.grad_accum, args.alpha, args.beta, args.w_identity, args.w_id_toxic, args.no_reweight)
        
        # [Fix] 所有 Rank 都参与验证，避免 NCCL 超时
        val_loss = evaluate(model, val_loader, device, args.w_identity, args.w_id_toxic, args.no_reweight)
        
        # 同步所有 Rank 的验证完成
        dist.barrier()
        
        # 只在主进程处理后续逻辑
        if is_main_process:
            if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(val_loss)
            loss_history["train"].append(train_loss); loss_history["val"].append(val_loss)
            print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), save_path)
            
            if early_stopping(val_loss): 
                print(f">>> [Early Stop] 提前结束。")
        
        # 同步早停信号
        stop_signal = torch.tensor(1 if (is_main_process and early_stopping.early_stop) else 0).to(device)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            break
    
    if is_main_process:
        with open(get_log_path(save_name + "_loss.json"), 'w') as f: json.dump(loss_history, f, indent=2)
        plt.figure(figsize=(10, 6)); plt.plot(range(1, len(loss_history["train"])+1), loss_history["train"], 'b-o', label='Train Loss')
        plt.plot(range(1, len(loss_history["val"])+1), loss_history["val"], 'r-o', label='Val Loss')
        plt.title(f'S2 Loss Curve: {save_name}'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150); plt.close()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
