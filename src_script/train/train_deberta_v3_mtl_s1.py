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
from sklearn.metrics import roc_auc_score

# =============================================================================
# ### 核心训练脚本：train_deberta_mtl_s1.py ###
# 改进点:
# 1. [P0] 开启 AMP 混合精度训练 (与 S2 一致)
# 2. [P0] Focal Loss alpha 1.0 (数据已1:1平衡，两类等权)
# 3. [P0] 删除无用的 SyncBatchNorm
# 4. [P1] AUC-based early stopping & checkpoint selection
# 5. [P1] Uncertainty Weighting 自动学习多任务权重
# 6. [P1] 评估时计算完整多任务 loss (与训练一致)
# 7. [P2] num_workers=4 提升数据加载速度
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model_deberta_v3_mtl import DebertaV3MTL
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping

from loss_functions import BCEFocalLoss

def uncertainty_weighted_loss(l_tox, l_sub, l_id, log_var_tox, log_var_sub, log_var_id):
    """
    Uncertainty Weighting (Kendall et al., 2018)
    loss_i = (1 / 2*sigma_i^2) * L_i + log(sigma_i)
    等价于: loss_i = L_i / (2 * exp(log_var)) + log_var / 2
    不确定性高的任务自动降权
    """
    w_tox = l_tox / (2 * torch.exp(log_var_tox)) + log_var_tox / 2
    w_sub = l_sub / (2 * torch.exp(log_var_sub)) + log_var_sub / 2
    w_id = l_id / (2 * torch.exp(log_var_id)) + log_var_id / 2
    return w_tox + w_sub + w_id

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps, only_toxicity=False, use_focal=True):
    model.train()

    if use_focal:
        criterion_tox = BCEFocalLoss(alpha=1.0, gamma=2.0)  # alpha=1.0: 数据已1:1平衡，两类等权
    else:
        criterion_tox = nn.BCEWithLogitsLoss()

    criterion_aux = nn.BCEWithLogitsLoss()

    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="[Train S1 AMP]")

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)

        with torch.cuda.amp.autocast():  # 开启 AMP (与 S2 保持一致)
            out = model(ids, mask)
            l_tox = criterion_tox(out['logits_tox'], y_tox)

            if only_toxicity:
                loss = l_tox
            else:
                l_sub = criterion_aux(out['logits_sub'], y_sub)
                l_id = criterion_aux(out['logits_id'], y_id)
                log_var_tox, log_var_sub, log_var_id = out['log_vars']
                loss = uncertainty_weighted_loss(l_tox, l_sub, l_id, log_var_tox, log_var_sub, log_var_id)

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

def evaluate(model, loader, device, use_focal=True):
    """
    评估函数: 计算完整多任务 loss + ROC-AUC
    改进: 评估 loss 包含辅助任务 (与训练一致), 并额外返回 AUC
    """
    model.eval()
    if use_focal:
        criterion_tox = BCEFocalLoss(alpha=1.0, gamma=2.0)
    else:
        criterion_tox = nn.BCEWithLogitsLoss()
    criterion_aux = nn.BCEWithLogitsLoss()

    total_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval S1]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            y_sub = batch['y_sub'].to(device)
            y_id = batch['y_id'].to(device)

            with torch.cuda.amp.autocast():
                out = model(ids, mask)
                l_tox = criterion_tox(out['logits_tox'], y_tox)
                l_sub = criterion_aux(out['logits_sub'], y_sub)
                l_id = criterion_aux(out['logits_id'], y_id)
                log_var_tox, log_var_sub, log_var_id = out['log_vars']
                loss = uncertainty_weighted_loss(l_tox, l_sub, l_id, log_var_tox, log_var_sub, log_var_id)

            total_loss += loss.item()

            probs = torch.sigmoid(out['logits_tox']).cpu().numpy().flatten()
            labels = (y_tox >= 0.5).int().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels)

    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, auc

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    parser = argparse.ArgumentParser(description="DeBERTa-V3 MTL Stage 1 (DDP)")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", -1))
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16, help="单卡BatchSize")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="linear")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42, help="数据采样种子(固定)")
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--no_pooling", action="store_true")
    parser.add_argument("--only_toxicity", action="store_true")
    parser.add_argument("--no_bar", action="store_true")

    # Ablation Flags
    parser.add_argument("--no_aug", action="store_true", help="Disables data augmentation")
    parser.add_argument("--no_focal", action="store_true", help="Disables Focal Loss (reverts to BCE)")
    parser.add_argument("--ablation_tag", type=str, default=None)

    args = parser.parse_args()

    # --- DDP Initialization ---
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    is_main_process = (local_rank == 0)

    if args.no_bar or (not is_main_process):
        global tqdm
        from tqdm import tqdm as _tqdm
        tqdm = lambda *args, **kwargs: _tqdm(*args, **kwargs, disable=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%m%d_%H%M")

    suffix = ""
    if args.no_pooling: suffix += "_NoPooling"
    if args.only_toxicity: suffix += "_OnlyTox"
    if args.no_aug: suffix += "_NoAug"
    if args.no_focal: suffix += "_NoFocal"
    if args.ablation_tag: suffix += f"_{args.ablation_tag}"

    save_name = f"DebertaV3MTL_S1{suffix}_Seed{args.seed}_Sample{args.sample_size}_{timestamp}"
    save_path = get_model_path(save_name + ".pth")

    if is_main_process:
        print(f"\n>>> 启动实验: {save_name} | Mode: DDP + AMP")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.data_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len, augment=not args.no_aug)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len, augment=False)

    # --- DDP Data Loading ---
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,       # 提升数据加载速度 (旧: 0)
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = DebertaV3MTL(args.model_name, use_attention_pooling=not args.no_pooling).to(device)

    # --- DDP Model Wrapping (删除无用的 SyncBatchNorm) ---
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=False)
    scaler = torch.cuda.amp.GradScaler()  # AMP GradScaler

    world_size = dist.get_world_size()
    num_steps = int(len(train_ds) / args.batch_size / world_size / args.accum_steps * args.epochs)
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True if is_main_process else False)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_steps), num_training_steps=num_steps)

    early_stopping = EarlyStopping(patience=args.early_patience)
    loss_history = {"train": [], "val": [], "val_auc": []}
    best_val_auc = 0.0  # 用 AUC 选 checkpoint (旧: 用 val_loss)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        if is_main_process: print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(
          model, train_loader, optimizer, scheduler, scaler, device, args.accum_steps,
          args.only_toxicity,
          use_focal=not args.no_focal
        )

        val_loss, val_auc = evaluate(model, val_loader, device, use_focal=not args.no_focal)

        dist.barrier()

        if is_main_process:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            loss_history["val_auc"].append(val_auc)

            # 打印 uncertainty weights
            try:
                log_vars = [p.item() for p in [model.module.log_var_tox, model.module.log_var_sub, model.module.log_var_id]]
                weights = [1.0 / (2 * np.exp(lv)) for lv in log_vars]
                print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
                print(f"  Task Weights -> Tox: {weights[0]:.3f} | Sub: {weights[1]:.3f} | Id: {weights[2]:.3f}")
            except Exception:
                print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            # 用 AUC 选 checkpoint (旧: 用 val_loss)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.module.state_dict(), save_path)
                print(f"  [Save] Best AUC={val_auc:.4f} -> {save_path}")

            if early_stopping(val_loss):
                print(f">>> [Early Stop] 提前结束。Best AUC={best_val_auc:.4f}")

        stop_signal = torch.tensor(1 if (is_main_process and early_stopping.early_stop) else 0).to(device)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            break

    if is_main_process:
        with open(get_log_path(save_name + "_loss.json"), 'w') as f:
            json.dump(loss_history, f, indent=2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(loss_history["train"]) + 1)
        ax1.plot(epochs_range, loss_history["train"], 'b-o', label='Train Loss')
        ax1.plot(epochs_range, loss_history["val"], 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch'); ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_range, loss_history["val_auc"], 'g-o', label='Val AUC')
        ax2.set_xlabel('Epoch'); ax2.set_title('Validation AUC'); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.suptitle(f'S1: {save_name}')
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150); plt.close()
        print(f">>> 实验完成。Best AUC={best_val_auc:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
