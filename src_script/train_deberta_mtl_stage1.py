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
# ### 核心训练脚本：train_deberta_mtl.py ###
# 设计说明：
# 本脚本执行 DebertaV3MTL 模型的第一阶段（基础多任务训练）。
# 严格遵循 200,000 条训练样本的对齐要求，并支持通过参数管理器配置。
# =============================================================================

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

# 离线环境变量设置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_deberta_mtl import DebertaV3MTL
from data_loader import ToxicityDataset, sample_aligned_data

def train_one_epoch(model, loader, optimizer, scheduler, device, accum_steps):
    """ 单个 Epoch 的训练逻辑 """
    model.train()
    # 使用 BCEWithLogitsLoss 进行分类监督，此时 output 没有经过 Sigmoid
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    pbar = tqdm(loader, desc="[Train]")
    
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)
        
        # 前向传播
        out = model(ids, mask)
        
        # 多任务损失加权：L = L_tox + 0.5 * L_sub + 0.2 * L_id
        # 该权重旨在让主任务占据主导，同时利用辅助任务进行偏见纠偏
        l_tox = criterion(out['logits_tox'], y_tox)
        l_sub = criterion(out['logits_sub'], y_sub)
        l_id = criterion(out['logits_id'], y_id)
        
        loss = l_tox + 0.5 * l_sub + 0.2 * l_id
        
        # 梯度累积
        loss = loss / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """ 验证集性能评估 """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            
            out = model(ids, mask)
            loss = criterion(out['logits_tox'], y_tox)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="DeBERTa-V3 Multi-Task Training (Stage 1)")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="预训练模型路径或名称")
    parser.add_argument("--sample_size", type=int, default=200000, help="统一训练样本量")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于数据对齐")
    parser.add_argument("--accum_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--max_len", type=int, default=256, help="序列最大长度")
    args = parser.parse_args()

    # 环境准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    # 结果保存路径（带时间戳和标识）
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_name = f"DebertaV3MTL_S1_Sample{args.sample_size}_{timestamp}.pth"
    save_path = os.path.join(BASE_DIR, "src_result", save_name)

    print(f"\n>>> 启动实验: {save_name}")
    print(f">>> 使用设备: {device} | 样本量: {args.sample_size} | Seed: {args.seed}")

    # 加载数据
    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    
    # 强制数据对齐
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    model = DebertaV3MTL(args.model_name).to(device)
    
    # 优化器与调度器
    num_steps = int(len(train_ds) / args.batch_size / args.accum_steps * args.epochs)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.accum_steps)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] 更优模型已保存至: {save_path}")

if __name__ == "__main__":
    main()
