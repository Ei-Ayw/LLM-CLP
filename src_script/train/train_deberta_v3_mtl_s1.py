import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast  # FP16 混合精度
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# =============================================================================
# ### 核心训练脚本：train_deberta_mtl.py ###
# 设计说明：
# 本脚本执行 DebertaV3MTL 模型的第一阶段（基础多任务训练）。
# 严格遵循 200,000 条训练样本的对齐要求，并支持通过参数管理器配置。
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
from train_utils import EarlyStopping

def train_one_epoch(model, loader, optimizer, scheduler, device, accum_steps, scaler, alpha, beta, only_toxicity=False):
    """
    单个 Epoch 的训练逻辑 (支持 FP16 混合精度)
    scaler: GradScaler 实例，用于 FP16 训练
    only_toxicity: 消融实验开关，True 时仅训练主任务
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(loader, desc="[Train FP16]")
    
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)
        
        # =====================================================================
        # FP16 混合精度训练：使用 autocast 自动管理精度
        # 优势：显存降低约 40%，训练速度提升 40-50%
        # =====================================================================
        with autocast():
            out = model(ids, mask)
            
            # 4.1 Loss: L = L_tox + 0.5 * L_sub + 0.2 * L_id
            l_tox = criterion(out['logits_tox'], y_tox)
            
            if only_toxicity:
                loss = l_tox
            else:
                l_sub = criterion(out['logits_sub'], y_sub)
                l_id = criterion(out['logits_id'], y_id)
                l_sub = criterion(out['logits_sub'], y_sub)
                l_id = criterion(out['logits_id'], y_id)
                loss = l_tox + alpha * l_sub + beta * l_id
        
        # FP16 梯度缩放
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
    parser.add_argument("--batch_size", type=int, default=48, help="3090 24G 可用 48+")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau", help="学习率调度策略")
    parser.add_argument("--patience", type=int, default=1, help="Plateau 调度器的耐心值")
    parser.add_argument("--early_patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数 (S1:6 + S2:4 = 10 epoch)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于数据对齐")
    parser.add_argument("--accum_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--max_len", type=int, default=256, help="序列最大长度")
    parser.add_argument("--alpha", type=float, default=0.5, help="子任务(Subtype)损失权重")
    parser.add_argument("--beta", type=float, default=0.2, help="子任务(Identity)损失权重")
    
    # 消融实验开关
    parser.add_argument("--no_pooling", action="store_true", help="消融实验：禁用 Attention Pooling，仅使用 [CLS]")
    parser.add_argument("--only_toxicity", action="store_true", help="消融实验：禁用 MTL，仅训练毒性主任务")
    
    args = parser.parse_args()

    # 环境准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    # 结果保存路径：根据消融开关动态命名以确保存储标识清晰
    timestamp = datetime.now().strftime("%m%d_%H%M")
    suffix = ""
    if args.no_pooling: suffix += "_NoPooling"
    if args.only_toxicity: suffix += "_OnlyTox"
    save_name = f"DebertaV3MTL_S1{suffix}_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_name + ".pth")

    print(f"\n>>> 启动实验: {save_name}")
    print(f">>> 使用设备: {device} | 样本量: {args.sample_size} | FP16: 启用")

    # 加载数据
    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    
    # 强制数据对齐
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化：通过参数开关控制是否启用 Attention Pooling
    model = DebertaV3MTL(args.model_name, use_attention_pooling=not args.no_pooling).to(device)
    
    # 优化器与调度器
    num_steps = int(len(train_ds) / args.batch_size / args.accum_steps * args.epochs)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
        print(">>> 启用自适应学习率: ReduceLROnPlateau")
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)
        print(">>> 启用线性预热学习率: Linear Schedule with Warmup")

    # =========================================================================
    # FP16 混合精度：GradScaler 用于动态缩放梯度，防止下溢
    # =========================================================================
    scaler = GradScaler()
    
    # 早停与记录
    early_stopping = EarlyStopping(patience=args.early_patience)
    loss_history = {"train": [], "val": []}
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.accum_steps, scaler, args.alpha, args.beta, args.only_toxicity)
        val_loss = evaluate(model, val_loader, device)
        val_loss = evaluate(model, val_loader, device)
        
        # 如果是 Plateau 调度器，在验证集后 Step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  [Scheduler] Adaptive LR Step. Current LR: {curr_lr}")
            
        # 记录 Loss
        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        
        print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] 更优模型已保存至: {save_path}")
            
        if early_stopping(val_loss):
            print(f">>> [Early Stop] 验证集 Loss 连续 {args.early_patience} 轮未优化，提前结束。")
            break
    
    # =========================================================================
    # 保存 Loss 历史 (JSON) 和生成曲线图 (PNG)
    # =========================================================================
    loss_json_path = get_log_path(save_name + "_loss.json")
    with open(loss_json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f">>> Loss 历史已保存: {loss_json_path}")
    
    # 生成 Loss 曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), loss_history["train"], 'b-o', label='Train Loss')
    plt.plot(range(1, args.epochs + 1), loss_history["val"], 'r-o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve: {save_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_fig_path = get_log_path(save_name + "_loss.png")
    plt.savefig(loss_fig_path, dpi=150)
    plt.close()
    print(f">>> Loss 曲线图已保存: {loss_fig_path}")

if __name__ == "__main__":
    main()

