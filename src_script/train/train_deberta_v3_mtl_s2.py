import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast  # FP16 混合精度
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# =============================================================================
# ### 核心训练脚本：train_deberta_mtl_stage2.py ###
# 设计说明：
# 本脚本执行 DebertaV3MTL 模型的第二阶段训练（身份感知加权微调）。
# 目的是通过对包含身份信息的样本进行重加权，缓解模型对特定群体的误判偏见。
# =============================================================================

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

# 离线环境变量设置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_deberta_v3_mtl import DebertaV3MTL
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path

def weighted_toxicity_loss(logits, targets, has_id, w_identity, w_id_toxic):
    """
    身份感知重加权损失函数。
    设计意图：
    - 对于“包含身份词但无毒”的样本（模型最易误判），增加权重 (w=2.5)。
    - 对于“包含身份词且有毒”的样本，适度增加权重 (w=1.5)。
    - 普通样本权重保持为 1.0。
    """
    weights = torch.ones_like(targets)
    has_id_mask = has_id.unsqueeze(-1).bool()
    is_toxic = targets >= 0.5
    
    # 场景1: 背景负样本包含身份词 (BPSN 相关) -> 权重 w_identity (默认 2.5)
    weights[(~is_toxic) & has_id_mask] = w_identity
    # 场景2: 背景正样本包含身份词 (BNSP 相关) -> 权重 w_id_toxic (默认 1.5)
    weights[is_toxic & has_id_mask] = w_id_toxic
    
    # 计算基础 BCE 损失（不进行均值聚合）
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(logits, targets)
    
    # 应用加权并手动计算均值
    return (loss * weights).mean()

def train_one_epoch(model, loader, optimizer, scheduler, device, accum_steps, scaler, alpha, beta, w_identity, w_id_toxic, no_reweight=False):
    """ 单个 Epoch 的重加权训练逻辑 (支持 FP16) """
    model.train()
    criterion_aux = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    pbar = tqdm(loader, desc="[Train S2 FP16]")
    
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)
        has_id = batch['has_id'].to(device)
        
        with autocast():
            out = model(ids, mask)
            
            if no_reweight:
                l_tox = criterion_aux(out['logits_tox'], y_tox)
            else:
                l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, w_identity, w_id_toxic)
            
            l_sub = criterion_aux(out['logits_sub'], y_sub)
            l_id = criterion_aux(out['logits_id'], y_id)
            loss = l_tox + alpha * l_sub + beta * l_id
        
        loss = loss / accum_steps
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """ 验证集评估 """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval S2]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            out = model(ids, mask)
            loss = criterion(out['logits_tox'], y_tox)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="DeBERTa-V3 Stage 2: Identity-Aware Reweighting")
    parser.add_argument("--s1_checkpoint", type=str, required=True, help="第一阶段训练好的权重路径")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小 (3090 24G + 梯度检查点 可用 32)")
    parser.add_argument("--lr", type=float, default=1e-5, help="第二阶段通常使用更小的学习率")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数 (S1:6 + S2:4 = 10 epoch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5, help="子任务(Subtype)损失权重")
    parser.add_argument("--beta", type=float, default=0.2, help="子任务(Identity)损失权重")
    parser.add_argument("--w_identity", type=float, default=2.5, help="身份负样本(Non-Toxic + Identity)重加权权重")
    parser.add_argument("--w_id_toxic", type=float, default=1.5, help="身份正样本(Toxic + Identity)重加权权重")
    parser.add_argument("--no_reweight", action="store_true", help="消融实验：禁用重加权逻辑 (w=1.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    # 结果保存路径：增加消融标识
    timestamp = datetime.now().strftime("%m%d_%H%M")
    suffix = "_NoReweight" if args.no_reweight else ""
    save_name = f"DebertaV3MTL_S2{suffix}_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_name + ".pth")

    print(f"\n>>> 启动 Stage 2 实验: {save_name}")
    print(f">>> 加载 Stage 1 权重: {args.s1_checkpoint} | FP16: 启用")

    # 加载数据
    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化并加载第一阶段权重
    model = DebertaV3MTL(args.model_name).to(device)
    if os.path.exists(args.s1_checkpoint):
        model.load_state_dict(torch.load(args.s1_checkpoint, map_location=device))
        print("  [Success] Stage 1 权重加载成功。")
    else:
        print(f"  [Warning] 未找到权重文件 {args.s1_checkpoint}，将从头开始训练。")

    num_steps = int(len(train_ds) / args.batch_size * args.epochs)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    # FP16 GradScaler
    scaler = GradScaler()
    
    # Loss 历史记录
    loss_history = {"train": [], "val": []}
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, accum_steps=2, scaler=scaler, 
                                     alpha=args.alpha, beta=args.beta, w_identity=args.w_identity, w_id_toxic=args.w_id_toxic,
                                     no_reweight=args.no_reweight)
        val_loss = evaluate(model, val_loader, device)
        
        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        
        print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] Stage 2 更优模型已保存至: {save_path}")
    
    # 保存 Loss 历史和曲线图
    loss_json_path = get_log_path(save_name + "_loss.json")
    with open(loss_json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), loss_history["train"], 'b-o', label='Train Loss')
    plt.plot(range(1, args.epochs + 1), loss_history["val"], 'r-o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Stage 2 Loss Curve: {save_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150)
    plt.close()
    print(f">>> Loss 曲线图已保存: {get_log_path(save_name + '_loss.png')}")

if __name__ == "__main__":
    main()
