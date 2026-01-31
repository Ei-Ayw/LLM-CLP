import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
# from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# =============================================================================
# ### 核心训练脚本：train_deberta_mtl.py (AMP加速版) ###
# 设计说明：
# 本脚本执行 DebertaV3MTL 模型的第一阶段（基础多任务训练）。
# 已针对 3090 24G 优化：启用 AMP、Gradient Checkpointing、短序列提速。
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

from loss_functions import BCEFocalLoss

def train_one_epoch(model, loader, optimizer, scheduler, device, accum_steps, alpha, beta, only_toxicity=False, use_focal=True):
    model.train()
    
    # 损失函数配置
    if use_focal:
        criterion_tox = BCEFocalLoss(alpha=12.5, gamma=2.0) # Focal Loss
    else:
        criterion_tox = nn.BCEWithLogitsLoss() # Standard BCE
        
    criterion_aux = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    optimizer.zero_grad() # [Fix] 显式初始化梯度，防止残留
    pbar = tqdm(loader, desc="[Train S1 AMP]")
    
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)
        
        # with torch.amp.autocast('cuda'): # [Safe Mode] 移除 AMP
        out = model(ids, mask)
        l_tox = criterion_tox(out['logits_tox'], y_tox)
        
        if only_toxicity:
            loss = l_tox
        else:
            l_sub = criterion_aux(out['logits_sub'], y_sub)
            l_id = criterion_aux(out['logits_id'], y_id)
            loss = l_tox + alpha * l_sub + beta * l_id
        
        loss = loss / accum_steps
        loss.backward() # [Safe Mode] 直接反向传播
        
        if (i + 1) % accum_steps == 0:
            optimizer.step() # [Safe Mode] 直接更新
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss() # Eval 仍用标准 Loss 看原始指标
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval S1]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            # with torch.amp.autocast('cuda'): # [Safe Mode]
            out = model(ids, mask)
            loss = criterion(out['logits_tox'], y_tox)
            total_loss += loss.item()
    return total_loss / len(loader)

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    parser = argparse.ArgumentParser(description="DeBERTa-V3 MTL Stage 1 (DDP)")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", -1)) # DDP 必需
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16, help="单卡BatchSize，DDP下设为16即可（总=4x16=64）")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1, help="MTL Weight for Subtypes (Downscaled from 0.5)")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--no_pooling", action="store_true")
    parser.add_argument("--only_toxicity", action="store_true")
    parser.add_argument("--no_bar", action="store_true")
    
    # Ablation Flags
    parser.add_argument("--no_aug", action="store_true", help="Disables data augmentation")
    parser.add_argument("--no_focal", action="store_true", help="Disables Focal Loss (reverts to BCE)")
    parser.add_argument("--ablation_tag", type=str, default=None, help="Tag for ablation study filename")
    
    args = parser.parse_args()

    # --- DDP Initialization ---
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    is_main_process = (local_rank == 0)

    # 静默模式处理 (仅主进程显示进度条，或者全部静默)
    if args.no_bar or (not is_main_process):
        global tqdm
        from tqdm import tqdm as _tqdm
        tqdm = lambda *args, **kwargs: _tqdm(*args, **kwargs, disable=True)

    # [Performance] Hardware Acceleration
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # 构建文件名后缀
    suffix = ""
    if args.no_pooling: suffix += "_NoPooling"
    if args.only_toxicity: suffix += "_OnlyTox"
    if args.no_aug: suffix += "_NoAug"
    if args.no_focal: suffix += "_NoFocal"
    if args.ablation_tag: suffix += f"_{args.ablation_tag}"
    
    save_name = f"DebertaV3MTL_S1{suffix}_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_name + ".pth")

    if is_main_process:
        print(f"\n>>> 启动实验: {save_name} | Mode: DDP")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len, augment=not args.no_aug) 
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len, augment=False)
    
    # --- DDP Data Loading ---
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=False, # Shuffle handled by sampler
        sampler=train_sampler,
        num_workers=0, 
        pin_memory=True
    )
    # 验证集简单处理，只让主进程验证，或者都验证但 sampler不shuffle
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    model = DebertaV3MTL(args.model_name, use_attention_pooling=not args.no_pooling).to(device)
    
    # --- DDP Model Wrapping ---
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model._set_static_graph()  # [Fix] 兼容 gradient checkpointing
    
    optimizer = AdamW(model.parameters(), lr=args.lr, fused=False)
    
    world_size = dist.get_world_size()
    num_steps = int(len(train_ds) / args.batch_size / world_size / args.accum_steps * args.epochs)
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True if is_main_process else False)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)

    early_stopping = EarlyStopping(patience=args.early_patience)
    loss_history = {"train": [], "val": []}
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch) # Shuffle for DDP
        
        if is_main_process: print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(
          model, train_loader, optimizer, scheduler, device, args.accum_steps, 
          args.alpha, args.beta, args.only_toxicity, 
          use_focal=not args.no_focal
        )
        
        # [Fix] 所有 Rank 都参与验证，避免 NCCL 超时
        val_loss = evaluate(model, val_loader, device)
        
        # 同步所有 Rank 的验证完成
        dist.barrier()
        
        # 只在主进程处理后续逻辑
        if is_main_process:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
                
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), save_path) # Save module only
                print(f"  [Save] 更优模型已保存至: {save_path}")
                
            if early_stopping(val_loss):
                print(f">>> [Early Stop] 提前结束。")
        
        # 同步早停信号
        stop_signal = torch.tensor(1 if (is_main_process and early_stopping.early_stop) else 0).to(device)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            break
    
    if is_main_process:
        with open(get_log_path(save_name + "_loss.json"), 'w') as f:
            json.dump(loss_history, f, indent=2)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history["train"]) + 1), loss_history["train"], 'b-o', label='Train Loss')
        plt.plot(range(1, len(loss_history["val"]) + 1), loss_history["val"], 'r-o', label='Val Loss')
        plt.xlabel('Epoch')
        plt.title(f'S1 Loss Curve: {save_name}')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150); plt.close()
        print(f">>> 实验完成。")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
