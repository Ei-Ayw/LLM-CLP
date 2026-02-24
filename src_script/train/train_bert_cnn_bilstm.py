import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.cuda.amp import GradScaler, autocast # Deprecated
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# =============================================================================
# ### 训练脚本：train_bert_cnn_bilstm.py ###
# 设计说明：
# 本脚本用于训练 BertCNNBiLSTM 混合架构模型。
# 支持自适应学习率 (ReduceLROnPlateau) 和早停机制 (EarlyStopping)。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model_bert_cnn_bilstm import BertCNNBiLSTM
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(loader, desc="[Train BertCNN]")
    for i, batch in enumerate(pbar):
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        y = batch['y_tox'].to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(): # 开启自动混合精度
            out = model(ids, mask)
            loss = criterion(out['logits_tox'], y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
            
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="BERT + CNN + BiLSTM Baseline Training (AMP)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=128, help="AMP+ShortLen 可稳定支持 128")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256, help="序列最大长度")
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--no_bar", action="store_true")
    args = parser.parse_args()

    # 静默模式处理 (for nohup)
    if args.no_bar:
        global tqdm
        from tqdm import tqdm as _tqdm
        tqdm = lambda *args, **kwargs: _tqdm(*args, **kwargs, disable=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [Performance] 开启硬件加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_basename = f"BertCNNBiLSTM_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_basename + ".pth")

    print(f"\n>>> 启动对比模型实验 (AMP): {save_basename}")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len, augment=True) # 增强
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len, augment=False)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    model = BertCNNBiLSTM(args.model_name).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training BertCNN...")
        model = nn.DataParallel(model)
        
    optimizer = AdamW(model.parameters(), lr=args.lr, fused=True)
    scaler = torch.cuda.amp.GradScaler() # 梯度缩放器
    
    num_steps = int(len(train_ds) / args.batch_size * args.epochs)
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)

    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=args.early_patience)
    loss_history = {"train": [], "val": []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                y = batch['y_tox'].to(device).unsqueeze(-1)
                with torch.cuda.amp.autocast():
                    out = model(ids, mask)
                    v_loss += nn.BCEWithLogitsLoss()(out['logits_tox'], y).item()
        val_loss = v_loss / len(val_loader)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        loss_history["train"].append(float(train_loss))
        loss_history["val"].append(float(val_loss))
        print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"  [Save] 更优模型已保存至: {save_path}")

        if early_stopping(val_loss):
            print(f">>> [Early Stop] 已触发。")
            break

    # 保存 Loss 历史和曲线图
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
    plt.tight_layout()
    plt.savefig(get_log_path(save_basename + "_loss.png"), dpi=150)
    plt.close()
    print(f">>> 实验完成。")

if __name__ == "__main__":
    main()
