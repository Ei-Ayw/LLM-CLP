import os
import sys

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
import torch.nn.functional as F
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
from torch.utils.data import Sampler
from sklearn.metrics import roc_auc_score

# =============================================================================
# HCMA 训练脚本: Hierarchical Conditional Metric-Aligned Debiasing
#
# 改进 1: Soft gate (sigmoid 替代 hard threshold)
# 改进 2: Slice-aware sampler (batch 内保证身份切片存在)
# 改进 3: Metric-aligned AUC surrogate loss
# 改进 4: Hierarchical conditional adversary (existence/coarse 保留, specific 对抗)
# =============================================================================

IDENTITY_COLS = [
    'male', 'female', 'homosexual_gay_or_lesbian',
    'christian', 'jewish', 'muslim',
    'black', 'white', 'psychiatric_or_mental_illness'
]

# 粗粒度分组映射: specific index → coarse index
# gender=0, sexual_orient=1, religion=2, race=3, disability=4
SPECIFIC_TO_COARSE = [0, 0, 1, 2, 2, 2, 3, 3, 4]
NUM_COARSE = 5

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_deberta_v3_hcma import DebertaV3HCMA
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping


# =============================================================================
# EMA
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
# 改进 2: Slice-Aware Sampler
# 保证每个 batch 中有足够的身份提及样本 (toxic + non-toxic)
# =============================================================================
class SliceAwareSampler(Sampler):
    """
    Batch 内分布控制:
      50% 随机自然分布
      25% non-toxic identity-mention
      15% toxic identity-mention
      10% toxic non-identity (hard negatives)
    """
    def __init__(self, dataset_df, batch_size, seed=42):
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.epoch = 0

        y_tox = dataset_df['y_tox_soft'].values if 'y_tox_soft' in dataset_df.columns else dataset_df['y_tox'].values
        is_toxic = y_tox >= 0.5
        has_id = dataset_df['has_identity'].values >= 1

        # 四个池子
        self.pool_bg = np.where(~is_toxic & ~has_id)[0]         # 非毒 + 无身份
        self.pool_id_nontox = np.where(~is_toxic & has_id)[0]   # 非毒 + 有身份 (去偏核心)
        self.pool_id_tox = np.where(is_toxic & has_id)[0]       # 有毒 + 有身份
        self.pool_tox_bg = np.where(is_toxic & ~has_id)[0]      # 有毒 + 无身份

        self.n_total = len(dataset_df)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed_for_epoch())
        bs = self.batch_size

        n_bg = max(1, int(bs * 0.50))
        n_id_nt = max(1, int(bs * 0.25))
        n_id_tx = max(1, int(bs * 0.15))
        n_tx_bg = bs - n_bg - n_id_nt - n_id_tx

        n_batches = self.n_total // bs
        indices = []

        for _ in range(n_batches):
            batch = []
            batch.extend(rng.choice(self.pool_bg, size=min(n_bg, len(self.pool_bg)), replace=True))
            batch.extend(rng.choice(self.pool_id_nontox, size=min(n_id_nt, len(self.pool_id_nontox)), replace=True))
            batch.extend(rng.choice(self.pool_id_tox, size=min(n_id_tx, len(self.pool_id_tox)), replace=True))
            if len(self.pool_tox_bg) > 0:
                batch.extend(rng.choice(self.pool_tox_bg, size=min(n_tx_bg, len(self.pool_tox_bg)), replace=True))
            else:
                batch.extend(rng.choice(self.pool_id_tox, size=min(n_tx_bg, len(self.pool_id_tox)), replace=True))
            indices.extend(batch[:bs])

        return iter(indices)

    def __len__(self):
        return (self.n_total // self.batch_size) * self.batch_size

    def seed_for_epoch(self):
        return self.rng.randint(0, 2**31) + self.epoch


# =============================================================================
# DDP-compatible Slice-Aware Sampler
# =============================================================================
class DistributedSliceAwareSampler(Sampler):
    """在 DDP 环境下包装 SliceAwareSampler，保证每个 rank 拿到不同子集"""
    def __init__(self, dataset_df, batch_size, num_replicas=None, rank=None, seed=42):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.inner = SliceAwareSampler(dataset_df, batch_size, seed=seed + rank)
        self.epoch = 0

        total = len(self.inner)
        self.num_samples = total // num_replicas
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.inner.set_epoch(epoch)

    def __iter__(self):
        all_indices = list(self.inner)
        all_indices = all_indices[:self.total_size]
        per_rank = len(all_indices) // self.num_replicas
        start = self.rank * per_rank
        return iter(all_indices[start:start + per_rank])

    def __len__(self):
        return self.num_samples


# =============================================================================
# 改进 3: Differentiable AUC Surrogate Loss
# =============================================================================
def pairwise_auc_loss(logits, targets, mask=None, n_pairs=512):
    """
    Pairwise logistic AUC surrogate:
      L = mean( log(1 + exp(-(s_pos - s_neg))) )
    """
    logits = logits.squeeze(-1)
    targets = targets.squeeze(-1)

    pos_mask = targets >= 0.5
    neg_mask = targets < 0.5
    if mask is not None:
        pos_mask = pos_mask & mask
        neg_mask = neg_mask & mask

    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # 随机采样 pairs
    n_pairs = min(n_pairs, len(pos_idx) * len(neg_idx))
    pi = pos_idx[torch.randint(0, len(pos_idx), (n_pairs,), device=logits.device)]
    ni = neg_idx[torch.randint(0, len(neg_idx), (n_pairs,), device=logits.device)]

    diff = logits[pi] - logits[ni]
    loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
    return loss


def compute_metric_aligned_loss(logits, targets, y_id, n_pairs=256):
    """
    Metric-aligned loss: Overall AUC + Subgroup/BPSN/BNSP AUC surrogates
    用 soft worst-group (log-sum-exp) 聚合逼近 PM₋₅
    """
    logits_flat = logits.squeeze(-1)
    targets_flat = targets.squeeze(-1)
    target_binary = targets_flat >= 0.5

    # 1. Overall AUC
    l_overall = pairwise_auc_loss(logits, targets, n_pairs=n_pairs)

    # 2. Per-subgroup AUC losses
    subgroup_losses = []
    bpsn_losses = []
    bnsp_losses = []

    for g in range(y_id.shape[1]):  # 9 个身份组
        g_mask = y_id[:, g] >= 0.5
        bg_mask = ~g_mask

        if g_mask.sum() < 4:
            continue

        # Subgroup AUC: 仅 subgroup 内的正负样本
        sub_loss = pairwise_auc_loss(logits, targets, mask=g_mask, n_pairs=n_pairs)
        if sub_loss.requires_grad:
            subgroup_losses.append(sub_loss)

        # BPSN: background positive + subgroup negative
        bpsn_pos = bg_mask & target_binary
        bpsn_neg = g_mask & ~target_binary
        if bpsn_pos.sum() >= 2 and bpsn_neg.sum() >= 2:
            bpsn_idx_p = torch.where(bpsn_pos)[0]
            bpsn_idx_n = torch.where(bpsn_neg)[0]
            np_ = min(n_pairs, len(bpsn_idx_p) * len(bpsn_idx_n))
            pi = bpsn_idx_p[torch.randint(0, len(bpsn_idx_p), (np_,), device=logits.device)]
            ni = bpsn_idx_n[torch.randint(0, len(bpsn_idx_n), (np_,), device=logits.device)]
            diff = logits_flat[pi] - logits_flat[ni]
            bpsn_losses.append(F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff)))

        # BNSP: background negative + subgroup positive
        bnsp_neg = bg_mask & ~target_binary
        bnsp_pos = g_mask & target_binary
        if bnsp_neg.sum() >= 2 and bnsp_pos.sum() >= 2:
            bnsp_idx_p = torch.where(bnsp_pos)[0]
            bnsp_idx_n = torch.where(bnsp_neg)[0]
            np_ = min(n_pairs, len(bnsp_idx_p) * len(bnsp_idx_n))
            pi = bnsp_idx_p[torch.randint(0, len(bnsp_idx_p), (np_,), device=logits.device)]
            ni = bnsp_idx_n[torch.randint(0, len(bnsp_idx_n), (np_,), device=logits.device)]
            diff = logits_flat[pi] - logits_flat[ni]
            bnsp_losses.append(F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff)))

    # 3. Soft worst-group 聚合 (log-sum-exp 逼近 PM₋₅)
    beta = 8.0  # 越大越接近 max (worst group)
    all_bias_losses = subgroup_losses + bpsn_losses + bnsp_losses
    if len(all_bias_losses) > 0:
        stacked = torch.stack(all_bias_losses)
        l_bias = (stacked * beta).logsumexp(0) / beta  # soft worst-group
    else:
        l_bias = torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Final: 0.25 * overall + 0.75 * bias (对齐 Jigsaw Final Metric 权重)
    l_metric = 0.25 * l_overall + 0.75 * l_bias
    return l_metric


# =============================================================================
# 改进 1: Soft Gate
# =============================================================================
def soft_gate(y_tox, y_ia, tau_tox=0.30, T_tox=0.08, tau_ia=0.20, T_ia=0.05):
    """
    Soft gate: toxicity 越低、identity_attack 越低 → g 越接近 1 (强去偏)
    g_i = sigmoid((τ_tox - y_tox) / T_tox) × sigmoid((τ_ia - y_ia) / T_ia)
    """
    g_tox = torch.sigmoid((tau_tox - y_tox) / T_tox)
    g_ia = torch.sigmoid((tau_ia - y_ia) / T_ia)
    return g_tox * g_ia


# =============================================================================
# 辅助: 构建 coarse 标签
# =============================================================================
def build_coarse_labels(y_id):
    """从 9 个 specific identity 标签构建 5 个 coarse group 标签"""
    batch_size = y_id.shape[0]
    y_coarse = torch.zeros(batch_size, NUM_COARSE, device=y_id.device)
    for spec_idx, coarse_idx in enumerate(SPECIFIC_TO_COARSE):
        y_coarse[:, coarse_idx] = torch.max(y_coarse[:, coarse_idx], y_id[:, spec_idx])
    return y_coarse


def build_exist_labels(y_id):
    """从 9 个 specific identity 标签构建 binary existence 标签"""
    return (y_id.max(dim=1, keepdim=True).values >= 0.5).float()


# =============================================================================
# Lambda Schedule
# =============================================================================
def lambda_schedule(step, total_steps, lambda_max, gamma):
    t = step / max(total_steps, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * t)) - 1.0)


# =============================================================================
# 身份分组差异化权重
# =============================================================================
IDENTITY_GROUP_WEIGHTS = [1.2, 1.0, 2.3, 1.8, 1.9, 2.9, 1.3, 2.6, 3.4]
_GROUP_WEIGHTS_TENSOR = None

def get_group_weights_tensor(device):
    global _GROUP_WEIGHTS_TENSOR
    if _GROUP_WEIGHTS_TENSOR is None or _GROUP_WEIGHTS_TENSOR.device != device:
        _GROUP_WEIGHTS_TENSOR = torch.tensor(IDENTITY_GROUP_WEIGHTS, dtype=torch.float, device=device)
    return _GROUP_WEIGHTS_TENSOR


def weighted_toxicity_loss(logits, targets, has_id, y_id, w_id_toxic=1.5, group_weights_tensor=None):
    weights = torch.ones_like(targets)
    has_id_mask = has_id.unsqueeze(-1).bool()
    is_toxic = targets >= 0.5
    if group_weights_tensor is not None:
        id_present = (y_id >= 0.5).float()
        per_sample_w = (id_present * group_weights_tensor.unsqueeze(0)).max(dim=1, keepdim=True).values.clamp(min=1.0)
        weights = torch.where((~is_toxic) & has_id_mask, per_sample_w, weights)
    else:
        weights[(~is_toxic) & has_id_mask] = 2.5
    weights[is_toxic & has_id_mask] = w_id_toxic
    loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
    return (loss * weights).mean()


# =============================================================================
# Phase A: Head 热身 (冻结 backbone)
# =============================================================================
def train_one_epoch_warmup(model, loader, optimizer, scaler, device, accum_steps,
                           w_id_toxic, aux_scale=0.3):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    gw = get_group_weights_tensor(device)
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
            out = model(ids, mask, lambda_adv=0.0)
            l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, y_id,
                                           w_id_toxic=w_id_toxic, group_weights_tensor=gw)
            l_sub = criterion(out['logits_sub'], y_sub)

            # 分层身份头 (Phase A: 正常梯度训练，λ=0 所以 GRL 不反转)
            y_exist = build_exist_labels(y_id)
            y_coarse = build_coarse_labels(y_id)
            l_exist = criterion(out['logits_id_exist'], y_exist)
            l_coarse = criterion(out['logits_id_coarse'], y_coarse)
            l_adv = criterion(out['logits_id_specific'], y_id)  # λ=0, 正常梯度预热

            loss = l_tox + aux_scale * l_sub + 0.1 * (l_exist + l_coarse + l_adv)

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
# Phase B: 对抗微调 (全参数 + GRL + Soft Gate + AUC Loss)
# =============================================================================
def train_one_epoch_adversarial(model, loader, optimizer, scheduler, scaler, device,
                                accum_steps, w_id_toxic, aux_scale, current_lambda,
                                ema=None, w_metric=0.30):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    criterion_nr = nn.BCEWithLogitsLoss(reduction='none')
    gw = get_group_weights_tensor(device)
    total_loss = total_l_tox = total_l_adv = total_l_metric = 0
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

            # 1. L_soft: 加权 BCE
            l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, y_id,
                                           w_id_toxic=w_id_toxic, group_weights_tensor=gw)
            l_sub = criterion(out['logits_sub'], y_sub)

            # 2. L_metric: AUC surrogate (改进 3)
            l_metric = compute_metric_aligned_loss(out['logits_tox'], y_tox, y_id, n_pairs=256)

            # 3. 分层身份 (改进 4)
            y_exist = build_exist_labels(y_id)
            y_coarse = build_coarse_labels(y_id)
            l_exist = criterion(out['logits_id_exist'], y_exist)
            l_coarse = criterion(out['logits_id_coarse'], y_coarse)

            # 4. Soft gate 对抗 (改进 1 + 改进 4)
            # 提取 identity_attack 列 (index=4 in subtype: severe,obscene,threat,insult,identity_attack,sexual)
            y_ia = y_sub[:, 4]  # identity_attack
            y_tox_flat = y_tox.squeeze(-1)
            gate = soft_gate(y_tox_flat, y_ia)  # (B,)

            adv_loss_per_sample = criterion_nr(out['logits_id_specific'], y_id).mean(dim=-1)
            l_adv = (gate * adv_loss_per_sample).mean()

            # 总损失
            loss = (1.0 - w_metric) * l_tox + w_metric * l_metric \
                   + aux_scale * l_sub \
                   + 0.05 * (l_exist + l_coarse) \
                   + l_adv

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
        total_l_metric += l_metric.item()
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}",
                         l_tox=f"{total_l_tox/(i+1):.4f}",
                         l_adv=f"{total_l_adv/(i+1):.4f}",
                         l_auc=f"{total_l_metric/(i+1):.4f}")

    n = len(loader)
    return total_loss / n, total_l_tox / n, total_l_adv / n, total_l_metric / n


# =============================================================================
# 评估
# =============================================================================
def evaluate(model, loader, device, w_id_toxic=1.5, aux_scale=0.3):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    gw = get_group_weights_tensor(device)
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
                                               w_id_toxic=w_id_toxic, group_weights_tensor=gw)
                l_sub = criterion(out['logits_sub'], y_sub)
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
    param_groups = []
    base_model = model.module if hasattr(model, 'module') else model

    # 1. Embeddings
    embed_params = list(base_model.deberta.embeddings.parameters())
    if embed_params:
        param_groups.append({"params": embed_params, "lr": base_lr * decay_factor ** 2})

    # 2. Encoder layers
    encoder_layers = base_model.deberta.encoder.layer
    n_layers = len(encoder_layers)
    mid_point = n_layers // 2
    for i, layer in enumerate(encoder_layers):
        lr = base_lr * decay_factor ** 2 if i < mid_point else base_lr * decay_factor
        param_groups.append({"params": list(layer.parameters()), "lr": lr})

    # 3. Heads + Projection (正常学习率)
    head_params = []
    for name, param in base_model.named_parameters():
        if any(k in name for k in ['projection', 'tox_head', 'subtype_head',
                                    'att_pooling', 'dropout',
                                    'id_exist_head', 'id_coarse_head']):
            head_params.append(param)
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    # 4. Adversary specific: 独立高学习率
    adv_params = list(base_model.adv_specific.parameters())
    if adv_params:
        param_groups.append({"params": adv_params, "lr": base_lr * adv_lr_mult})

    return param_groups


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HCMA: Hierarchical Conditional Metric-Aligned Debiasing")
    parser.add_argument("--s1_checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--w_id_toxic", type=float, default=1.5)
    parser.add_argument("--no_bar", action="store_true")

    # Phase A
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_lr", type=float, default=1e-4)

    # Phase B
    parser.add_argument("--adv_epochs", type=int, default=6)
    parser.add_argument("--adv_lr", type=float, default=2e-6)
    parser.add_argument("--lambda_max", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--aux_scale", type=float, default=0.15)
    parser.add_argument("--w_metric", type=float, default=0.30, help="AUC surrogate loss weight")
    parser.add_argument("--layer_decay", type=float, default=0.95)
    parser.add_argument("--adv_lr_mult", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default=None)

    args = parser.parse_args()

    # --- DDP ---
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        is_main_process = (args.local_rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    if args.no_bar or not is_main_process:
        global tqdm
        from tqdm import tqdm as _tqdm
        tqdm = lambda *a, **kw: _tqdm(*a, **kw, disable=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    save_name = f"HCMA_Seed{args.seed}_Sample{args.sample_size}_{datetime.now().strftime('%m%d_%H%M')}"
    save_path = get_model_path(save_name + ".pth")

    # --- Data ---
    data_dir = args.data_dir if args.data_dir else os.path.join(BASE_DIR, "data")
    train_df = pd.read_parquet(os.path.join(data_dir, "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.data_seed)

    if is_main_process:
        n_tox = (train_df['y_tox'] >= 0.5).sum() if 'y_tox' in train_df.columns else (train_df['y_tox_soft'] >= 0.5).sum()
        n_id = train_df['has_identity'].sum()
        print(f"[Data] Train: {len(train_df):,} | Toxic: {n_tox:,} ({n_tox/len(train_df)*100:.1f}%) | HasID: {n_id:,} ({n_id/len(train_df)*100:.1f}%)")
        print(f"[Data] Val: {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)

    # Slice-aware sampler (改进 2)
    train_sampler = DistributedSliceAwareSampler(
        train_df, batch_size=args.batch_size,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        seed=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # --- Model ---
    model = DebertaV3HCMA(args.model_name).to(device)

    # 加载 S1 checkpoint
    if os.path.exists(args.s1_checkpoint):
        state_dict = torch.load(args.s1_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main_process:
            print(f"  [S1 Load] missing: {len(missing)} keys")
            print(f"  [S1 Load] unexpected: {len(unexpected)} keys")
    else:
        raise FileNotFoundError(f"S1 checkpoint not found: {args.s1_checkpoint}")

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True)
    base_model = model.module if hasattr(model, 'module') else model

    # =========================================================================
    # Phase A: Head 热身
    # =========================================================================
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Phase A: Head Warmup ({args.warmup_epochs} epochs, backbone frozen)")
        print(f"{'='*60}")

    for param in base_model.deberta.parameters():
        param.requires_grad = False

    warmup_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_a = AdamW(warmup_params, lr=args.warmup_lr, weight_decay=args.weight_decay)
    scaler_a = torch.cuda.amp.GradScaler()

    loss_history = {
        "phase_a_train": [], "phase_a_val": [], "phase_a_auc": [],
        "phase_b_train": [], "phase_b_val": [], "phase_b_auc": [],
        "phase_b_final": [], "phase_b_l_adv": [], "phase_b_l_metric": [], "phase_b_lambda": [],
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
    # Phase B: 对抗微调
    # =========================================================================
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Phase B: HCMA Fine-tuning ({args.adv_epochs} epochs)")
        print(f"  lambda_max={args.lambda_max}, w_metric={args.w_metric}, aux_scale={args.aux_scale}")
        print(f"{'='*60}")

    for param in base_model.deberta.parameters():
        param.requires_grad = True

    param_groups = build_layer_wise_param_groups(model, base_lr=args.adv_lr,
                                                  decay_factor=args.layer_decay,
                                                  adv_lr_mult=args.adv_lr_mult)
    optimizer_b = AdamW(param_groups, weight_decay=args.weight_decay)
    scaler_b = torch.cuda.amp.GradScaler()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    num_steps = int(len(train_ds) / args.batch_size / world_size / args.grad_accum * args.adv_epochs)
    warmup_steps = int(args.warmup_ratio * num_steps)
    scheduler_b = get_cosine_schedule_with_warmup(optimizer_b, num_warmup_steps=warmup_steps,
                                                  num_training_steps=num_steps)

    ema = ModelEMA(base_model, decay=args.ema_decay) if args.ema_decay > 0 else None
    best_final = 0.0
    early_stopping = EarlyStopping(patience=args.early_patience)

    for epoch in range(args.adv_epochs):
        train_sampler.set_epoch(epoch + args.warmup_epochs)

        epoch_progress = (epoch + 0.5) / args.adv_epochs
        current_lambda = lambda_schedule(
            step=int(epoch_progress * num_steps),
            total_steps=num_steps,
            lambda_max=args.lambda_max,
            gamma=args.gamma,
        )

        if is_main_process:
            print(f"\n[Phase B] Epoch {epoch+1}/{args.adv_epochs} | λ={current_lambda:.4f}")

        train_loss, l_tox_avg, l_adv_avg, l_metric_avg = train_one_epoch_adversarial(
            model, train_loader, optimizer_b, scheduler_b, scaler_b, device,
            args.grad_accum, args.w_id_toxic, args.aux_scale,
            current_lambda=current_lambda, ema=ema, w_metric=args.w_metric,
        )

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
            loss_history["phase_b_l_metric"].append(l_metric_avg)
            loss_history["phase_b_lambda"].append(current_lambda)

            print(f"  Train Loss: {train_loss:.4f} | L_tox: {l_tox_avg:.4f} | L_adv: {l_adv_avg:.4f} | L_auc: {l_metric_avg:.4f}")
            print(f"  [Final={final_metrics['final']:.4f}] Bias={final_metrics['bias_score']:.4f} | AUC={final_metrics['overall_auc']:.4f}")
            print(f"  PM(Sub)={final_metrics['pm_sub']:.4f} | PM(BPSN)={final_metrics['pm_bpsn']:.4f} | PM(BNSP)={final_metrics['pm_bnsp']:.4f}")
            if ema is not None:
                print(f"  [EMA] decay={args.ema_decay}")

            if final_metrics['final'] > best_final:
                best_final = final_metrics['final']
                torch.save(base_model.state_dict(), save_path)
                print(f"  [Save] Best Final={final_metrics['final']:.4f} -> {save_path}")

            if early_stopping(val_loss):
                print(f">>> [Early Stop] Best Final={best_final:.4f}")

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

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        if loss_history["phase_a_train"]:
            ep_a = range(1, len(loss_history["phase_a_train"]) + 1)
            axes[0, 0].plot(ep_a, loss_history["phase_a_train"], 'b-o', label='Train')
            axes[0, 0].plot(ep_a, loss_history["phase_a_val"], 'r-o', label='Val')
            axes[0, 0].set_title('Phase A: Loss')
            axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        if loss_history["phase_b_train"]:
            ep_b = range(1, len(loss_history["phase_b_train"]) + 1)
            axes[0, 1].plot(ep_b, loss_history["phase_b_train"], 'b-o', label='Train')
            axes[0, 1].plot(ep_b, loss_history["phase_b_val"], 'r-o', label='Val')
            axes[0, 1].set_title('Phase B: Loss')
            axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

            axes[0, 2].plot(ep_b, loss_history["phase_b_l_metric"], 'c-o', label='L_auc')
            axes[0, 2].set_title('Phase B: AUC Surrogate Loss')
            axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

            ax_adv = axes[1, 0]
            ax_adv.plot(ep_b, loss_history["phase_b_l_adv"], 'g-o', label='L_adv')
            ax_adv.set_title('Phase B: Adv Loss & Lambda')
            ax_lambda = ax_adv.twinx()
            ax_lambda.plot(ep_b, loss_history["phase_b_lambda"], 'm--s', label='λ')
            ax_adv.legend(loc='upper left'); ax_lambda.legend(loc='upper right')
            ax_adv.grid(True, alpha=0.3)

            axes[1, 1].plot(ep_b, loss_history["phase_b_final"], 'k-o', label='Final')
            axes[1, 1].set_title('Phase B: Final Metric')
            axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

            axes[1, 2].plot(ep_b, loss_history["phase_b_auc"], 'purple', marker='o', label='Val AUC')
            axes[1, 2].set_title('Phase B: Val AUC')
            axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'HCMA: {save_name}')
        plt.tight_layout()
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150)
        plt.close()

        print(f"\n>>> 实验完成。Best Final={best_final:.4f}")
        print(f">>> Checkpoint: {save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
