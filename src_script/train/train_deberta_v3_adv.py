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
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score

# =============================================================================
# ### 对抗去偏训练脚本: train_deberta_v3_adv.py ###
# Phase A: Head 热身 (冻结 backbone, 训头部)
# Phase B: 对抗微调 (GRL sigmoid ramp-up)
# =============================================================================

IDENTITY_COLS = [
    'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian',
    'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
]

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_deberta_v3_adversarial import DebertaV3Adversarial
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])


# =============================================================================
# 身份分组差异化权重 (对数平滑逆频率加权)
# =============================================================================
IDENTITY_GROUP_WEIGHTS = [1.2, 1.0, 2.3, 1.8, 1.9, 2.9, 1.3, 2.6, 3.4]
_GROUP_WEIGHTS_TENSOR = None


def get_group_weights_tensor(device):
    global _GROUP_WEIGHTS_TENSOR
    if _GROUP_WEIGHTS_TENSOR is None or _GROUP_WEIGHTS_TENSOR.device != device:
        _GROUP_WEIGHTS_TENSOR = torch.tensor(IDENTITY_GROUP_WEIGHTS, dtype=torch.float, device=device)
    return _GROUP_WEIGHTS_TENSOR


def weighted_toxicity_loss(logits, targets, has_id, y_id, w_id_toxic=1.5, group_weights_tensor=None):
    """身份感知加权损失函数"""
    weights = torch.ones_like(targets)
    has_id_mask = has_id.unsqueeze(-1).bool()
    is_toxic = targets >= 0.5

    if group_weights_tensor is not None:
        id_present = (y_id >= 0.5).float()
        per_sample_w = (id_present * group_weights_tensor.unsqueeze(0)).max(dim=1, keepdim=True).values
        per_sample_w = per_sample_w.clamp(min=1.0)
        weights = torch.where((~is_toxic) & has_id_mask, per_sample_w, weights)
    else:
        weights[(~is_toxic) & has_id_mask] = 2.5

    weights[is_toxic & has_id_mask] = w_id_toxic

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(logits, targets)
    return (loss * weights).mean()


# =============================================================================
# Lambda Sigmoid Ramp-up: λ(t) = λ_max × (2/(1+exp(-γt))-1)
# =============================================================================
def lambda_schedule(step, total_steps, lambda_max, gamma):
    """Sigmoid ramp-up schedule for adversarial lambda"""
    t = step / max(total_steps, 1)  # normalized progress [0, 1]
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * t)) - 1.0)


# =============================================================================
# Phase A: Head 热身 (冻结 backbone)
# =============================================================================
def train_one_epoch_warmup(model, loader, optimizer, scaler, device, accum_steps,
                           w_id_toxic, aux_scale=0.3):
    """Phase A: λ=0, 冻结 backbone, 只训 projection + heads"""
    model.train()
    criterion_aux = nn.BCEWithLogitsLoss()
    gw_tensor = get_group_weights_tensor(device)
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="[Phase A Warmup]")

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)
        has_id = batch['has_id'].to(device)

        with torch.cuda.amp.autocast():
            out = model(ids, mask, lambda_adv=0.0)  # no adversarial gradient
            l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, y_id,
                                           w_id_toxic=w_id_toxic,
                                           group_weights_tensor=gw_tensor)
            l_sub = criterion_aux(out['logits_sub'], y_sub)
            # Phase A: 也训练 adv_head (用正常梯度，因为 λ=0 时 GRL 不反转)
            l_adv = criterion_aux(out['logits_id_adv'], y_id)
            loss = l_tox + aux_scale * l_sub + 0.1 * l_adv  # 小权重预热 adv_head

        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")

    return total_loss / len(loader)


# =============================================================================
# Phase B: 对抗微调 (全参数 + GRL)
# =============================================================================
def train_one_epoch_adversarial(model, loader, optimizer, scheduler, scaler, device,
                                accum_steps, w_id_toxic, aux_scale, current_lambda,
                                ema=None):
    """Phase B: 全参数训练 + 条件 adversarial loss (只对非毒样本)"""
    model.train()
    criterion_aux = nn.BCEWithLogitsLoss()
    criterion_adv = nn.BCEWithLogitsLoss(reduction='none')
    gw_tensor = get_group_weights_tensor(device)
    total_loss = 0
    total_l_tox = 0
    total_l_adv = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"[Phase B λ={current_lambda:.4f}]")

    for i, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_tox = batch['y_tox'].to(device).unsqueeze(-1)
        y_sub = batch['y_sub'].to(device)
        y_id = batch['y_id'].to(device)
        has_id = batch['has_id'].to(device)

        with torch.cuda.amp.autocast():
            out = model(ids, mask, lambda_adv=current_lambda)
            l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, y_id,
                                           w_id_toxic=w_id_toxic,
                                           group_weights_tensor=gw_tensor)
            l_sub = criterion_aux(out['logits_sub'], y_sub)

            # 条件 GRL: 只对非毒样本计算 adversarial loss
            # 有毒样本保留身份信息 (用于检测仇恨言论)
            # 非毒样本反转身份梯度 (打破"提到身份→非毒"的捷径)
            non_toxic_mask = (y_tox.squeeze(-1) < 0.5)
            if non_toxic_mask.any():
                adv_loss_per_sample = criterion_adv(out['logits_id_adv'], y_id)
                # 对每个样本的9个身份取均值，得到 per-sample loss
                adv_loss_per_sample = adv_loss_per_sample.mean(dim=-1)
                # 只保留非毒样本的 loss
                l_adv = adv_loss_per_sample[non_toxic_mask].mean()
            else:
                l_adv = torch.tensor(0.0, device=device)

            loss = l_tox + aux_scale * l_sub + l_adv

        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model.module if hasattr(model, 'module') else model)

        total_loss += loss.item() * accum_steps
        total_l_tox += l_tox.item()
        total_l_adv += l_adv.item()
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}",
                         l_tox=f"{total_l_tox/(i+1):.4f}",
                         l_adv=f"{total_l_adv/(i+1):.4f}")

    n = len(loader)
    return total_loss / n, total_l_tox / n, total_l_adv / n


# =============================================================================
# 评估函数
# =============================================================================
def evaluate(model, loader, device, w_id_toxic=1.5, aux_scale=0.3):
    """评估: 计算 loss + AUC"""
    model.eval()
    criterion_aux = nn.BCEWithLogitsLoss()
    gw_tensor = get_group_weights_tensor(device)
    total_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            has_id = batch['has_id'].to(device)
            y_id = batch['y_id'].to(device)
            y_sub = batch['y_sub'].to(device)

            with torch.cuda.amp.autocast():
                out = model(ids, mask, lambda_adv=0.0)
                l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, y_id,
                                               w_id_toxic=w_id_toxic,
                                               group_weights_tensor=gw_tensor)
                l_sub = criterion_aux(out['logits_sub'], y_sub)
                loss = l_tox + aux_scale * l_sub

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


def power_mean(values, p=-5):
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    arr = np.clip(arr, 1e-10, None)
    return float(np.power(np.mean(np.power(arr, p)), 1.0 / p))


def evaluate_with_final_metric(model, loader, device, val_df):
    """评估: Overall AUC + BiasScore + Final Metric"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval Final]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                out = model(ids, mask, lambda_adv=0.0)
            probs = torch.sigmoid(out['logits_tox']).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(batch['y_tox'].numpy().flatten())

    probs_arr = np.array(all_probs)
    labels_binary = (np.array(all_labels) >= 0.5).astype(int)
    try:
        overall_auc = roc_auc_score(labels_binary, probs_arr)
    except ValueError:
        overall_auc = 0.5

    eval_df = val_df.copy().iloc[:len(probs_arr)]
    eval_df['_prob'] = probs_arr
    pm_aucs = {'subgroup': [], 'bpsn': [], 'bnsp': []}
    for col in IDENTITY_COLS:
        sub_mask = eval_df[col] >= 0.5
        if sub_mask.sum() == 0:
            continue
        target = (eval_df['y_tox_soft'] if 'y_tox_soft' in eval_df.columns else eval_df['y_tox']) >= 0.5
        sub_target = target[sub_mask]
        if sub_target.nunique() >= 2:
            pm_aucs['subgroup'].append(roc_auc_score(sub_target, eval_df[sub_mask]['_prob']))
        bpsn_mask = (sub_mask & ~target) | (~sub_mask & target)
        bpsn_target = target[bpsn_mask]
        if bpsn_target.nunique() >= 2:
            pm_aucs['bpsn'].append(roc_auc_score(bpsn_target, eval_df[bpsn_mask]['_prob']))
        bnsp_mask = (sub_mask & target) | (~sub_mask & ~target)
        bnsp_target = target[bnsp_mask]
        if bnsp_target.nunique() >= 2:
            pm_aucs['bnsp'].append(roc_auc_score(bnsp_target, eval_df[bnsp_mask]['_prob']))

    pm_sub = power_mean(pm_aucs['subgroup']) if pm_aucs['subgroup'] else 0.5
    pm_bpsn = power_mean(pm_aucs['bpsn']) if pm_aucs['bpsn'] else 0.5
    pm_bnsp = power_mean(pm_aucs['bnsp']) if pm_aucs['bnsp'] else 0.5
    bias_score = (pm_sub + pm_bpsn + pm_bnsp) / 3.0
    final_metric = 0.25 * overall_auc + 0.75 * bias_score
    return {
        'final': final_metric, 'bias_score': bias_score, 'overall_auc': overall_auc,
        'pm_sub': pm_sub, 'pm_bpsn': pm_bpsn, 'pm_bnsp': pm_bnsp,
    }


# =============================================================================
# Layer-wise LR decay
# =============================================================================
def build_layer_wise_param_groups(model, base_lr, decay_factor=0.95, adv_lr_mult=5.0):
    """
    Layer-wise learning rate decay for adversarial model
    底层 lr 更小，顶层 + heads 用 base_lr
    adv_discriminator 用 adv_lr_mult × base_lr (独立更高学习率)
    """
    param_groups = []
    base_model = model.module if hasattr(model, 'module') else model

    # 1. Embeddings: 最小 lr
    embed_params = list(base_model.deberta.embeddings.parameters())
    if embed_params:
        param_groups.append({"params": embed_params, "lr": base_lr * decay_factor ** 2})

    # 2. Encoder layers: 渐进式 lr
    encoder_layers = base_model.deberta.encoder.layer
    n_layers = len(encoder_layers)
    mid_point = n_layers // 2
    for i, layer in enumerate(encoder_layers):
        layer_params = list(layer.parameters())
        if i < mid_point:
            lr = base_lr * decay_factor ** 2
        else:
            lr = base_lr * decay_factor
        param_groups.append({"params": layer_params, "lr": lr})

    # 3. Heads + Projection: base_lr
    head_params = []
    for name, param in base_model.named_parameters():
        if any(k in name for k in ['projection', 'tox_head', 'subtype_head',
                                    'att_pooling', 'dropout']):
            head_params.append(param)
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    # 4. Adversary discriminator: 独立更高学习率 (追上 backbone)
    adv_params = list(base_model.adv_discriminator.parameters())
    if adv_params:
        param_groups.append({"params": adv_params, "lr": base_lr * adv_lr_mult})

    return param_groups


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DeBERTa-V3 Adversarial Debiasing")
    parser.add_argument("--s1_checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--w_id_toxic", type=float, default=1.5)
    parser.add_argument("--no_bar", action="store_true")

    # Phase A
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--warmup_lr", type=float, default=1e-4)

    # Phase B
    parser.add_argument("--adv_epochs", type=int, default=4)
    parser.add_argument("--adv_lr", type=float, default=2e-6)
    parser.add_argument("--lambda_max", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--aux_scale", type=float, default=0.3)
    parser.add_argument("--layer_decay", type=float, default=0.95)
    parser.add_argument("--adv_lr_mult", type=float, default=5.0, help="对抗头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录 (含 train/val_processed.parquet)")

    args = parser.parse_args()

    # --- DDP Initialization ---
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        is_main_process = (args.local_rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

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

    save_name = f"DebertaV3Adv_Seed{args.seed}_Sample{args.sample_size}_{datetime.now().strftime('%m%d_%H%M')}"
    save_path = get_model_path(save_name + ".pth")

    # --- Data ---
    data_dir = args.data_dir if args.data_dir else os.path.join(BASE_DIR, "data")
    train_df = pd.read_parquet(os.path.join(data_dir, "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.data_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # --- Model ---
    model = DebertaV3Adversarial(args.model_name).to(device)

    # 加载 S1 checkpoint (strict=False: 忽略 identity_head, 随机初始化 adv_discriminator)
    if os.path.exists(args.s1_checkpoint):
        state_dict = torch.load(args.s1_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main_process:
            print(f"  [S1 Load] missing keys: {missing}")
            print(f"  [S1 Load] unexpected keys: {unexpected}")
            print(f"  [S1 Load] 加载成功 (strict=False)")
    else:
        raise FileNotFoundError(f"S1 checkpoint not found: {args.s1_checkpoint}")

    # --- DDP ---
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True)
    base_model = model.module if hasattr(model, 'module') else model

    # =========================================================================
    # Phase A: Head 热身 (冻结 backbone, 只训 projection + heads)
    # =========================================================================
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Phase A: Head Warmup ({args.warmup_epochs} epochs, backbone frozen)")
        print(f"{'='*60}")

    # 冻结 backbone
    for param in base_model.deberta.parameters():
        param.requires_grad = False

    # Phase A optimizer: 只训可学习参数
    warmup_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_a = AdamW(warmup_params, lr=args.warmup_lr, weight_decay=args.weight_decay)
    scaler_a = torch.cuda.amp.GradScaler()

    loss_history = {
        "phase_a_train": [], "phase_a_val": [], "phase_a_auc": [],
        "phase_b_train": [], "phase_b_val": [], "phase_b_auc": [],
        "phase_b_final": [], "phase_b_l_adv": [], "phase_b_lambda": [],
    }

    for epoch in range(args.warmup_epochs):
        train_sampler.set_epoch(epoch)
        if is_main_process:
            print(f"\n[Phase A] Epoch {epoch+1}/{args.warmup_epochs}")

        train_loss = train_one_epoch_warmup(
            model, train_loader, optimizer_a, scaler_a, device,
            args.grad_accum, args.w_id_toxic, args.aux_scale,
        )
        val_loss, val_auc = evaluate(model, val_loader, device, args.w_id_toxic, args.aux_scale)

        dist.barrier()

        if is_main_process:
            loss_history["phase_a_train"].append(train_loss)
            loss_history["phase_a_val"].append(val_loss)
            loss_history["phase_a_auc"].append(val_auc)
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

    # =========================================================================
    # Phase B: 对抗微调 (解冻全部, GRL ramp-up)
    # =========================================================================
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Phase B: Adversarial Fine-tuning ({args.adv_epochs} epochs)")
        print(f"  lambda_max={args.lambda_max}, gamma={args.gamma}, aux_scale={args.aux_scale}")
        print(f"{'='*60}")

    # 解冻 backbone
    for param in base_model.deberta.parameters():
        param.requires_grad = True

    # Phase B optimizer: layer-wise LR decay
    param_groups = build_layer_wise_param_groups(model, base_lr=args.adv_lr, decay_factor=args.layer_decay, adv_lr_mult=args.adv_lr_mult)
    optimizer_b = AdamW(param_groups, weight_decay=args.weight_decay)
    scaler_b = torch.cuda.amp.GradScaler()

    # Cosine scheduler
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    num_steps = int(len(train_ds) / args.batch_size / world_size / args.grad_accum * args.adv_epochs)
    warmup_steps = int(args.warmup_ratio * num_steps)
    scheduler_b = get_cosine_schedule_with_warmup(optimizer_b, num_warmup_steps=warmup_steps,
                                                  num_training_steps=num_steps)

    # EMA
    ema = ModelEMA(base_model, decay=args.ema_decay) if args.ema_decay > 0 else None

    best_final = 0.0
    early_stopping = EarlyStopping(patience=args.early_patience)

    # 计算每个 epoch 的 lambda
    total_adv_steps = num_steps  # total optimizer steps in Phase B

    for epoch in range(args.adv_epochs):
        train_sampler.set_epoch(epoch + args.warmup_epochs)

        # Lambda for this epoch (use epoch midpoint for stable per-epoch lambda)
        epoch_progress = (epoch + 0.5) / args.adv_epochs
        current_lambda = lambda_schedule(
            step=int(epoch_progress * total_adv_steps),
            total_steps=total_adv_steps,
            lambda_max=args.lambda_max,
            gamma=args.gamma,
        )

        if is_main_process:
            print(f"\n[Phase B] Epoch {epoch+1}/{args.adv_epochs} | λ={current_lambda:.4f}")

        train_loss, l_tox_avg, l_adv_avg = train_one_epoch_adversarial(
            model, train_loader, optimizer_b, scheduler_b, scaler_b, device,
            args.grad_accum, args.w_id_toxic, args.aux_scale,
            current_lambda=current_lambda, ema=ema,
        )

        # 评估: 如果有 EMA，用 EMA 权重
        if ema is not None:
            ema.apply_shadow(base_model)

        val_loss, val_auc = evaluate(model, val_loader, device, args.w_id_toxic, args.aux_scale)
        final_metrics = evaluate_with_final_metric(model, val_loader, device, val_df)

        dist.barrier()

        if is_main_process:
            loss_history["phase_b_train"].append(train_loss)
            loss_history["phase_b_val"].append(val_loss)
            loss_history["phase_b_auc"].append(val_auc)
            loss_history["phase_b_final"].append(final_metrics['final'])
            loss_history["phase_b_l_adv"].append(l_adv_avg)
            loss_history["phase_b_lambda"].append(current_lambda)

            print(f"  Train Loss: {train_loss:.4f} | L_tox: {l_tox_avg:.4f} | L_adv: {l_adv_avg:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            print(f"  [Final={final_metrics['final']:.4f}] BiasScore={final_metrics['bias_score']:.4f}")
            print(f"  PM(Sub)={final_metrics['pm_sub']:.4f} | PM(BPSN)={final_metrics['pm_bpsn']:.4f} | PM(BNSP)={final_metrics['pm_bnsp']:.4f}")
            if ema is not None:
                print(f"  [EMA] decay={args.ema_decay}")

            # Checkpoint 选择: Final Metric
            if final_metrics['final'] > best_final:
                best_final = final_metrics['final']
                torch.save(base_model.state_dict(), save_path)
                print(f"  [Save] Best Final={final_metrics['final']:.4f} -> {save_path}")

            if early_stopping(val_loss):
                print(f">>> [Early Stop] Best Final={best_final:.4f}")

        # 恢复非EMA权重继续训练
        if ema is not None:
            ema.restore(base_model)

        stop_signal = torch.tensor(1 if (is_main_process and early_stopping.early_stop) else 0).to(device)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            break

    # --- Save & Plot ---
    if is_main_process:
        with open(get_log_path(save_name + "_loss.json"), 'w') as f:
            json.dump(loss_history, f, indent=2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Phase A loss
        if loss_history["phase_a_train"]:
            ep_a = range(1, len(loss_history["phase_a_train"]) + 1)
            axes[0, 0].plot(ep_a, loss_history["phase_a_train"], 'b-o', label='Train')
            axes[0, 0].plot(ep_a, loss_history["phase_a_val"], 'r-o', label='Val')
            axes[0, 0].set_title('Phase A: Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Phase B loss
        if loss_history["phase_b_train"]:
            ep_b = range(1, len(loss_history["phase_b_train"]) + 1)
            axes[0, 1].plot(ep_b, loss_history["phase_b_train"], 'b-o', label='Train')
            axes[0, 1].plot(ep_b, loss_history["phase_b_val"], 'r-o', label='Val')
            axes[0, 1].set_title('Phase B: Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Phase B: L_adv + lambda
        if loss_history["phase_b_l_adv"]:
            ax_adv = axes[1, 0]
            ax_adv.plot(ep_b, loss_history["phase_b_l_adv"], 'g-o', label='L_adv')
            ax_adv.set_ylabel('L_adv')
            ax_adv.set_title('Phase B: Adversarial Loss & Lambda')
            ax_lambda = ax_adv.twinx()
            ax_lambda.plot(ep_b, loss_history["phase_b_lambda"], 'm--s', label='λ')
            ax_lambda.set_ylabel('λ')
            ax_adv.legend(loc='upper left')
            ax_lambda.legend(loc='upper right')
            ax_adv.grid(True, alpha=0.3)

        # Phase B: Final Metric
        if loss_history["phase_b_final"]:
            axes[1, 1].plot(ep_b, loss_history["phase_b_final"], 'k-o', label='Final Metric')
            axes[1, 1].axhline(y=0.9365, color='orange', linestyle='--', label='Vanilla baseline')
            axes[1, 1].set_title('Phase B: Final Metric')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Adversarial Debiasing: {save_name}')
        plt.tight_layout()
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150)
        plt.close()

        print(f"\n>>> 实验完成。Best Final={best_final:.4f}")
        print(f">>> Checkpoint: {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
