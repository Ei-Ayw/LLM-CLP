import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# =============================================================================
# ### 训练脚本：train_vanilla_deberta_v3.py (DDP & AMP加速版) ###
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model_vanilla_deberta_v3 import VanillaDeBERTaV3
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(loader, desc="[Train DeBERTa]")
    for i, batch in enumerate(pbar):
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        y = batch['y_tox'].to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        # [Performance] Disable explicit AMP if stability is an issue, 
        # but here we follow the baseline pattern. 
        # Note: DeBERTaV3 often prefers bf16 orfp32 for stability.
        with torch.amp.autocast('cuda'):
            out = model(ids, mask)
            loss = criterion(out['logits_tox'], y)
        
        # We don't use GradScaler here to be consistent with MTL script which marked it [Safe Mode]
        # But for vanilla, we can use it if we want. Standardizing to MTL's [Safe Mode] for now.
        loss.backward()
        optimizer.step()
        
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval DeBERTa]"):
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y = batch['y_tox'].to(device).unsqueeze(-1)
            out = model(ids, mask)
            loss = criterion(out['logits_tox'], y)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Vanilla DeBERTa-v3 Training (DDP)")
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--no_bar", action="store_true")
    args = parser.parse_args()

    # --- DDP Initialization ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    is_main_process = (local_rank == 0)

    # 静默模式处理
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
    save_basename = f"VanillaDeBERTa_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_basename + ".pth")

    if is_main_process:
        print(f"\n>>> 启动实验: {save_basename} | Mode: DDP")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)
    
    # --- DDP Data Loading ---
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=0, 
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    model = VanillaDeBERTaV3(args.model_path).to(device)
    
    # --- DDP Model Wrapping ---
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    world_size = dist.get_world_size()
    num_steps = int(len(train_ds) / args.batch_size / world_size * args.epochs)
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True if is_main_process else False)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)

    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=args.early_patience)
    loss_history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if is_main_process: print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        
        # 同步验证
        dist.barrier()
        
        if is_main_process:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            loss_history["train"].append(float(train_loss))
            loss_history["val"].append(float(val_loss))
            print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), save_path)
                # [Fix] 同时保存 Tokenizer 到同级目录以供评估使用
                tokenizer.save_pretrained(os.path.join(MODEL_DIR, save_basename + "_tokenizer"))
                print(f"  [Save] 更优模型与 Tokenizer 已保存至: {save_path}")

            if early_stopping(val_loss):
                print(f">>> [Early Stop] 已触发。")
        
        # 同步早停
        stop_signal = torch.tensor(1 if (is_main_process and early_stopping.early_stop) else 0).to(device)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            break

    # 保存结果 (仅主进程)
    if is_main_process:
        loss_json_path = get_log_path(save_basename + "_loss.json")
        with open(loss_json_path, 'w') as f:
            json.dump(loss_history, f, indent=2)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history["train"]) + 1), loss_history["train"], 'b-o', label='Train Loss')
        plt.plot(range(1, len(loss_history["val"]) + 1), loss_history["val"], 'r-o', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve: {save_basename}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(get_log_path(save_basename + "_loss.png"), dpi=150)
        plt.close()
        print(f">>> 实验完成。")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
