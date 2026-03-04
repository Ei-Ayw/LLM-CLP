"""
=============================================================================
IACD: Identity-Agnostic Contrastive Debiasing
File 3/3: 完整训练脚本 (DDP + AMP)
=============================================================================
用法:
  # E1: IACD-Full (完整方案)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.run \
      --nproc_per_node=7 --master_port=29500 \
      src_script/train/train_deberta_v3_iacd.py \
      --seed 42 --alpha 0.3 --temperature 0.07 --cross_weight 0.5

  # E2: Standard SupCon (消融: 去掉跨身份加权)
  ... --no_cross_weight

  # E3: BCE-Only (消融: 去掉对比损失 = Vanilla DeBERTa 微调)
  ... --no_contrastive
=============================================================================
"""
import os
import sys

# [CRITICAL] 必须在 import transformers 之前设置所有 HF 环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# =============================================================================
# 项目路径
# =============================================================================
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_and_loss import DebertaV3IACD, FairSupConLoss
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping
from utils_ddp import diff_all_gather


# =============================================================================
# Training
# =============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    criterion_cls, criterion_con, alpha, use_contrastive):
    model.train()
    total_loss, total_cls, total_con = 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc="[Train IACD]")
    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_soft = batch['y_tox'].to(device).unsqueeze(-1)      # soft label [0,1]
        y_id = batch['y_id'].to(device)                       # identity [B, 9]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(ids, mask, return_proj=use_contrastive)

            # (1) 分类损失: BCE with soft labels
            l_cls = criterion_cls(out['logits_tox'], y_soft)

            # (2) 对比损失: FairSupCon (可选)
            if use_contrastive and criterion_con is not None:
                z = out['z']
                y_bin = (y_soft.squeeze(-1) >= 0.5).float()
                id_bin = (y_id >= 0.5).float()

                # DDP: 可微分 all_gather -> 全局 batch 上计算对比损失
                z_all = diff_all_gather(z)
                y_bin_all = diff_all_gather(y_bin)
                id_bin_all = diff_all_gather(id_bin)

                l_con = criterion_con(z_all, y_bin_all, id_bin_all)
            else:
                l_con = torch.tensor(0.0, device=device)

            loss = l_cls + alpha * l_con

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_cls += l_cls.item()
        total_con += l_con.item()
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}",
                         cls=f"{total_cls/(i+1):.4f}",
                         con=f"{total_con/(i+1):.4f}")

    n = len(loader)
    return total_loss / n, total_cls / n, total_con / n


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval IACD]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_soft = batch['y_tox'].to(device).unsqueeze(-1)

            with torch.cuda.amp.autocast():
                out = model(ids, mask, return_proj=False)
                loss = criterion(out['logits_tox'], y_soft)

            total_loss += loss.item()
            probs = torch.sigmoid(out['logits_tox']).cpu().numpy().flatten()
            labels = (y_soft.squeeze(-1) >= 0.5).int().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return avg_loss, auc


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="IACD: Identity-Agnostic Contrastive Debiasing (DDP)")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", -1))
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=32, help="单卡 BatchSize")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--early_patience", type=int, default=3)

    # IACD 超参数
    parser.add_argument("--alpha", type=float, default=0.05, help="对比损失权重")
    parser.add_argument("--temperature", type=float, default=0.07, help="对比学习温度 τ")
    parser.add_argument("--cross_weight", type=float, default=0.5, help="跨身份加权 λ")

    # 消融开关
    parser.add_argument("--no_contrastive", action="store_true", help="E3: 去掉对比损失 (= Vanilla BCE)")
    parser.add_argument("--no_cross_weight", action="store_true", help="E2: 去掉跨身份加权 (= Standard SupCon)")
    parser.add_argument("--ablation_tag", type=str, default=None)
    parser.add_argument("--no_bar", action="store_true")
    args = parser.parse_args()

    # --- DDP 初始化 ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    is_main = (local_rank == 0)
    world_size = dist.get_world_size()

    if args.no_bar or not is_main:
        global tqdm
        from tqdm import tqdm as _tqdm
        tqdm = lambda *a, **kw: _tqdm(*a, **kw, disable=True)

    # --- 随机种子 ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # --- 实验命名 ---
    suffix = ""
    if args.no_contrastive:
        suffix += "_BCEOnly"
    elif args.no_cross_weight:
        suffix += "_StdCon"
    else:
        suffix += f"_a{args.alpha}_t{args.temperature}_lam{args.cross_weight}"
    if args.ablation_tag:
        suffix += f"_{args.ablation_tag}"

    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_name = f"IACD{suffix}_Seed{args.seed}_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_name + ".pth")

    if is_main:
        print(f"\n{'='*60}")
        print(f">>> IACD 实验启动: {save_name}")
        print(f">>> DDP: {world_size} GPUs | AMP: ON")
        if args.no_contrastive:
            print(f">>> 模式: BCE-Only (E3 消融)")
        elif args.no_cross_weight:
            print(f">>> 模式: Standard SupCon (E2 消融)")
        else:
            print(f">>> 模式: IACD-Full (α={args.alpha}, τ={args.temperature}, λ={args.cross_weight})")
        print(f"{'='*60}\n")

    # --- 数据加载 ---
    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.data_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)

    # DDP 标准采样器 (使用全量数据, 对比损失在全局 batch 上自然获取身份多样性)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # --- 模型 ---
    use_contrastive = not args.no_contrastive
    model = DebertaV3IACD(args.model_name).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=not use_contrastive)  # projector unused when no_contrastive

    # --- 优化器 + 调度器 ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    num_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_steps),
        num_training_steps=num_steps
    )

    # --- 损失函数 ---
    criterion_cls = nn.BCEWithLogitsLoss()

    if use_contrastive:
        cw = 0.0 if args.no_cross_weight else args.cross_weight
        criterion_con = FairSupConLoss(
            temperature=args.temperature,
            cross_weight=cw
        )
    else:
        criterion_con = None

    alpha = args.alpha if use_contrastive else 0.0

    # --- 训练循环 ---
    early_stopping = EarlyStopping(patience=args.early_patience)
    best_val_auc = 0.0
    history = {"train_loss": [], "train_cls": [], "train_con": [],
               "val_loss": [], "val_auc": []}

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # DDP: 确保每个 epoch 不同的 shuffle

        if is_main:
            print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_cls, train_con = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            criterion_cls, criterion_con, alpha, use_contrastive
        )

        val_loss, val_auc = evaluate(model, val_loader, device)

        dist.barrier()

        if is_main:
            history["train_loss"].append(train_loss)
            history["train_cls"].append(train_cls)
            history["train_con"].append(train_con)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            print(f"  Train -> Total: {train_loss:.4f} | BCE: {train_cls:.4f} | Con: {train_con:.4f}")
            print(f"  Val   -> Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.module.state_dict(), save_path)
                print(f"  [Save] Best AUC={val_auc:.4f} -> {save_path}")

            # EarlyStopping 基于 AUC (取负值，因为 EarlyStopping 监控 loss 下降)
            if early_stopping(-val_auc):
                print(f">>> [Early Stop] Best AUC={best_val_auc:.4f}")

        # 同步 early stop 信号
        stop_signal = torch.tensor(
            1 if (is_main and early_stopping.early_stop) else 0
        ).to(device)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            break

    # --- 保存日志 + 绘图 ---
    if is_main:
        log_json = get_log_path(save_name + "_loss.json")
        with open(log_json, 'w') as f:
            json.dump(history, f, indent=2)

        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs_range, history["train_loss"], 'b-o', label='Train Total')
        axes[0].plot(epochs_range, history["val_loss"], 'r-o', label='Val Loss')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, history["train_cls"], 'g-o', label='BCE Loss')
        axes[1].plot(epochs_range, history["train_con"], 'm-o', label='Contrastive Loss')
        axes[1].set_title('Loss Components'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs_range, history["val_auc"], 'c-o', label='Val AUC')
        axes[2].set_title('Validation AUC'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'IACD: {save_name}')
        plt.tight_layout()
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150)
        plt.close()

        print(f"\n{'='*60}")
        print(f">>> 实验完成。Best AUC={best_val_auc:.4f}")
        print(f">>> 权重: {save_path}")
        print(f">>> 日志: {log_json}")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
