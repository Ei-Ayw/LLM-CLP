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
# ### 训练脚本：train_vanilla_roberta.py ###
# 设计说明：
# 用于运行原生 roberta-base 的基准实验。
# 遵循“一模型一脚本”规范，独立管理 RoBERTa 的实验生命周期。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_vanilla_roberta import VanillaRoBERTa
from exp_data_loader import ToxicityDataset, sample_aligned_data

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
    save_path = os.path.join(BASE_DIR, "src_result", f"VanillaRoBERTa_Sample{args.sample_size}_{timestamp}.pth")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = VanillaRoBERTa(args.model_path).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_steps = int(len(train_ds) / args.batch_size * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, device, 1)
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
