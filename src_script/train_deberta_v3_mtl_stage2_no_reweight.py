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

# =============================================================================
# ### 训练脚本：train_deberta_v3_mtl_stage2_no_reweight.py ###
# 设计说明：
# 针对 Ablation-3 的物理对齐脚本。
# 它是 Stage 2 的克隆版，唯一的物理区别是它显式地将损失权重保持为 w=1.0（不加权）。
# 为了遵循一模型一脚本的标识性，我们在此创建独立的物理文件。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR); sys.path.append(os.path.join(BASE_DIR, "src_model")); sys.path.append(os.path.join(BASE_DIR, "src_script"))

os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_deberta_v3_mtl import DebertaV3MTL
from data_loader import ToxicityDataset, sample_aligned_data

def train_one_epoch(model, loader, optimizer, scheduler, device, accum_steps):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(loader, desc="[Train S2 NoReweight]")
    for i, batch in enumerate(pbar):
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub, y_id = batch['y_sub'].to(device), batch['y_id'].to(device)
        out = model(ids, mask)
        
        # 物理消融：此处强制不使用加权损失函数
        l_tox = criterion(out['logits_tox'], y_tox)
        l_sub = criterion(out['logits_sub'], y_sub)
        l_id = criterion(out['logits_id'], y_id)
        
        loss = (l_tox + 0.5 * l_sub + 0.2 * l_id) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1_checkpoint", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(BASE_DIR, "src_result", f"DebertaV3MTL_S2_NoReweight_Sample{args.sample_size}_{datetime.now().strftime('%m%d_%H%M')}.pth")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", local_files_only=True)
    train_ds = ToxicityDataset(train_df, tokenizer)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = DebertaV3MTL("microsoft/deberta-v3-base").to(device)
    model.load_state_dict(torch.load(args.s1_checkpoint, map_location=device))

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_steps = int(len(train_ds) / args.batch_size * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    for epoch in range(args.epochs):
        train_one_epoch(model, loader, optimizer, scheduler, device, 2)
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
