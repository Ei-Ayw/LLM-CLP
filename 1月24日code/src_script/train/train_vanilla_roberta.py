import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# =============================================================================
# ### 训练脚本：train_vanilla_roberta.py ###
# 设计说明：
# 用于运行原生 roberta-base 的基准实验。
# 遵循“一模型一脚本”规范，独立管理 RoBERTa 的实验生命周期。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_vanilla_roberta import VanillaRoBERTa
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path

def train_one_epoch(model, loader, optimizer, scheduler, device, accum_steps):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(loader, desc="[Train RoBERTa]")
    for i, batch in enumerate(pbar):
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        y = batch['y_tox'].to(device).unsqueeze(-1)
        out = model(ids, mask)
        loss = criterion(out['logits_tox'], y) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Vanilla RoBERTa Training")
    parser.add_argument("--model_path", type=str, default="roberta-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数 (统一为 10 epoch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_basename = f"VanillaRoBERTa_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_basename + ".pth")

    print(f"\n>>> 启动原生 RoBERTa 实验: {save_basename}")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = VanillaRoBERTa(args.model_path).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_steps = int(len(train_ds) / args.batch_size * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    best_val_loss = float('inf')
    loss_history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, 1)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                y = batch['y_tox'].to(device).unsqueeze(-1)
                out = model(ids, mask)
                v_loss += nn.BCEWithLogitsLoss()(out['logits_tox'], y).item()
        val_loss = v_loss / len(val_loader)
        
        loss_history["train"].append(float(train_loss))
        loss_history["val"].append(float(val_loss))
        print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] 更优模型已保存至: {save_path}")

    # 保存 Loss 历史和曲线图
    loss_json_path = get_log_path(save_basename + "_loss.json")
    with open(loss_json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), loss_history["train"], 'b-o', label='Train Loss')
    plt.plot(range(1, args.epochs + 1), loss_history["val"], 'r-o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve: {save_basename}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_log_path(save_basename + "_loss.png"), dpi=150)
    plt.close()
    print(f">>> 实验完成，模型与日志已保存。")

if __name__ == "__main__":
    main()
