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
# ### 训练脚本：train_bert_cnn_bilstm.py ###
# 设计说明：
# 本脚本用于训练 BertCNNBiLSTM 混合架构模型。
# 作为本文的重要对比实验，该模型验证了在 BERT 之后增加卷积和循环神经网络对毒性分类的贡献。
# =============================================================================

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

# 离线环境变量设置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_bert_cnn_bilstm import BertCNNBiLSTM
from exp_data_loader import ToxicityDataset, sample_aligned_data

def train_one_epoch(model, loader, optimizer, scheduler, device, accum_steps):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(loader, desc="[Train BertCNN]")
    
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        
        out = model(ids, mask)
        loss = criterion(out['logits_tox'], y_tox)
        
        loss = loss / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
        
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="BERT + CNN + BiLSTM Baseline Training")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_name = f"BertCNNBiLSTM_Sample{args.sample_size}_{timestamp}.pth"
    save_path = os.path.join(BASE_DIR, "src_result", save_name)

    print(f"\n>>> 启动对比模型实验: {save_name}")

    # 加载数据与对齐
    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    model = BertCNNBiLSTM(args.model_name).to(device)
    
    num_steps = int(len(train_ds) / args.batch_size * args.epochs)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, accum_steps=1)
        
        # 简单评估
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                y = batch['y_tox'].to(device).unsqueeze(-1)
                out = model(ids, mask)
                v_loss += nn.BCEWithLogitsLoss()(out['logits_tox'], y).item()
        val_loss = v_loss / len(val_loader)
        
        print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] 更优模型已保存至: {save_path}")

if __name__ == "__main__":
    main()
