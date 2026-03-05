"""
=============================================================================
### 公平性训练脚本: train_deberta_v3_fair_s2.py ###
论文核心方法实现 — 在原有 MTL 架构上演进:
  (1) 原版 MTL: 3任务共享backbone → 发现梯度冲突，AUC 被辅助任务拖累
  (2) 本脚本: 冻结辅助头，引入 Slice Ranking Loss + Anchor-PCGrad + CLP
      → 指标对齐 + 方向控制 + 反事实公平 三合一

核心组件:
  A. SliceRankingLoss: 对齐 Subgroup/BPSN/BNSP AUC 的 pairwise ranking 代理损失
  B. Anchor-PCGrad:    主任务梯度锚定，公平梯度投影去冲突
  C. CLP:             反事实 Logit Pairing，消除身份词捷径
  D. 三段 Schedule:    Warmup(只主任务) → Debias(+公平) → Stabilize(固定)
  E. Final Metric:     0.25*Overall_AUC + 0.75*BiasScore 做 checkpoint 选择
=============================================================================
"""
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse, json, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from contextlib import nullcontext
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_deberta_v3_mtl import DebertaV3MTL
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping

IDENTITY_COLS = [
    'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian',
    'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
]

# =============================================================================
# 身份分组差异化权重 (对数平滑逆频率加权)
# 顺序与 IDENTITY_COLS 一致
# =============================================================================
IDENTITY_GROUP_WEIGHTS = [1.2, 1.0, 2.3, 1.8, 1.9, 2.9, 1.3, 2.6, 3.4]
_GROUP_WEIGHTS_TENSOR = None

def get_group_weights_tensor(device):
    global _GROUP_WEIGHTS_TENSOR
    if _GROUP_WEIGHTS_TENSOR is None or _GROUP_WEIGHTS_TENSOR.device != device:
        _GROUP_WEIGHTS_TENSOR = torch.tensor(IDENTITY_GROUP_WEIGHTS, dtype=torch.float, device=device)
    return _GROUP_WEIGHTS_TENSOR

def weighted_toxicity_loss(logits, targets, has_id, y_id, w_id_toxic=1.5, group_weights_tensor=None):
    """
    身份感知加权损失: 对 has_identity & non-toxic 样本按身份稀有度差异化加权,
    对 has_identity & toxic 样本固定加权 w_id_toxic.
    """
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
# EMA (复用)
# =============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])

# =============================================================================
# A. Slice-based Pairwise Ranking Loss
#    直接对齐 Borkan et al. 的 Subgroup/BPSN/BNSP AUC 定义
# =============================================================================
class SliceRankingLoss(nn.Module):
    """
    对每个 identity group 构造三类 pair (Subgroup/BPSN/BNSP),
    用 pairwise logistic ranking loss 做可微 AUC surrogate.
    聚合方式: power-mean (p>1, 越大越关注最差组).
    """
    def __init__(self, num_groups=9, power_p=4, max_pairs_per_type=256):
        super().__init__()
        self.num_groups = num_groups
        self.power_p = power_p
        self.max_pairs = max_pairs_per_type

    def _pairwise_logistic(self, pos_logits, neg_logits):
        """log(1 + exp(-(s_pos - s_neg))), 对所有 pair 取均值"""
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return None
        # 高效: 不做全笛卡尔积, 而是采样
        n_pos, n_neg = pos_logits.size(0), neg_logits.size(0)
        n_pairs = min(n_pos * n_neg, self.max_pairs)
        if n_pos * n_neg <= self.max_pairs:
            # 全 pair
            diffs = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)  # [n_pos, n_neg]
            return F.softplus(-diffs).mean()
        else:
            # 随机采样 pair
            idx_p = torch.randint(0, n_pos, (n_pairs,), device=pos_logits.device)
            idx_n = torch.randint(0, n_neg, (n_pairs,), device=neg_logits.device)
            diffs = pos_logits[idx_p] - neg_logits[idx_n]
            return F.softplus(-diffs).mean()

    def forward(self, logits_tox, y_tox, y_id):
        """
        Args:
            logits_tox: [B, 1] toxicity logits
            y_tox: [B, 1] toxicity labels (soft)
            y_id: [B, 9] identity labels
        Returns:
            aggregated slice ranking loss (scalar)
        """
        logits = logits_tox.squeeze(-1)  # [B]
        is_toxic = (y_tox.squeeze(-1) >= 0.5)
        group_losses = []

        for g in range(self.num_groups):
            in_group = (y_id[:, g] >= 0.5)
            not_in_group = ~in_group

            # 四类样本索引
            sub_pos = in_group & is_toxic      # subgroup & toxic
            sub_neg = in_group & (~is_toxic)   # subgroup & non-toxic
            bg_pos = not_in_group & is_toxic   # background & toxic
            bg_neg = not_in_group & (~is_toxic) # background & non-toxic

            type_losses = []

            # Subgroup AUC: sub_pos vs sub_neg
            l = self._pairwise_logistic(logits[sub_pos], logits[sub_neg])
            if l is not None: type_losses.append(l)

            # BPSN AUC: bg_pos vs sub_neg (身份词误报场景)
            l = self._pairwise_logistic(logits[bg_pos], logits[sub_neg])
            if l is not None: type_losses.append(l)

            # BNSP AUC: sub_pos vs bg_neg (身份词漏报场景)
            l = self._pairwise_logistic(logits[sub_pos], logits[bg_neg])
            if l is not None: type_losses.append(l)

            if type_losses:
                group_losses.append(torch.stack(type_losses).mean())

        if not group_losses:
            return torch.tensor(0.0, device=logits_tox.device, requires_grad=True)

        # Power-mean 聚合: 强调最差组
        losses = torch.stack(group_losses)
        if self.power_p <= 1:
            return losses.mean()
        return torch.pow(torch.mean(torch.pow(losses, self.power_p)), 1.0 / self.power_p)

# =============================================================================
# B. Anchor-PCGrad: 主任务梯度锚定投影
# =============================================================================
def anchor_pcgrad_step(model, l_main, l_bias, lambda_bias, optimizer, scaler, accum_steps, ema=None):
    """
    Anchor-PCGrad: 保护主任务梯度方向, 投影公平梯度的冲突分量.

    g_final = g_main + λ * project(g_bias, g_main)

    其中 project: 如果 g_bias 与 g_main 冲突(点积<0), 切掉对抗分量.
    """
    # 收集可训练参数名
    param_names = [n for n, p in model.named_parameters() if p.requires_grad]

    no_sync_ctx = model.no_sync() if isinstance(model, DDP) else nullcontext()

    with no_sync_ctx:
        # 1. 主任务梯度
        model.zero_grad()
        scaler.scale(l_main / accum_steps).backward(retain_graph=True)
        g_main = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g_main[n] = p.grad.detach().clone()

        # 2. 公平损失梯度
        model.zero_grad()
        scaler.scale(l_bias / accum_steps).backward()
        g_bias = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g_bias[n] = p.grad.detach().clone()

    # 3. Anchor-PCGrad 投影 (只对 backbone 参数)
    #    backbone = 所有非 head 参数
    backbone_keys = [n for n in param_names
                     if not any(k in n.replace('module.', '') for k in
                                ['proj_tox', 'tox_head', 'att_pooling', 'log_var', 'drop_'])]

    dot = torch.tensor(0.0, device=next(iter(g_main.values())).device)
    norm_sq = torch.tensor(0.0, device=dot.device)
    for key in backbone_keys:
        if key in g_bias and key in g_main:
            dot = dot + (g_bias[key] * g_main[key]).sum()
            norm_sq = norm_sq + (g_main[key] ** 2).sum()

    was_conflict = False
    if dot.item() < 0 and norm_sq.item() > 0:
        coef = dot / (norm_sq + 1e-12)
        for key in backbone_keys:
            if key in g_bias and key in g_main:
                g_bias[key] = g_bias[key] - coef * g_main[key]
        was_conflict = True

    # 4. 合并: g_final = g_main + λ * g_bias_projected
    model.zero_grad()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = g_main.get(n, None)
        gb = g_bias.get(n, None)
        if g is not None and gb is not None:
            p.grad = g + lambda_bias * gb
        elif g is not None:
            p.grad = g
        elif gb is not None:
            p.grad = lambda_bias * gb

    # 5. DDP all-reduce
    if dist.is_initialized():
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(dist.get_world_size())

    return was_conflict

# =============================================================================
# C. CLP (Counterfactual Logit Pairing) 近似
#    不做文本替换, 而是约束同 toxicity-label 的不同 identity 样本 logit 一致
# =============================================================================
def compute_clp_loss(logits_tox, y_tox, y_id, has_id):
    """
    Batch 内 CLP 近似: 对 identity-mention 且同 toxicity-label 的样本对,
    惩罚 logit 差异. 重点约束 identity & non-toxic (BPSN 误报场景).
    """
    id_mask = has_id.bool()
    if id_mask.sum() < 2:
        return torch.tensor(0.0, device=logits_tox.device, requires_grad=True)

    logits = logits_tox[id_mask].squeeze(-1)
    labels = (y_tox[id_mask].squeeze(-1) >= 0.5).float()
    n = logits.size(0)

    # 同标签 mask
    same_label = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # 非毒性样本对权重更高 (BPSN 场景)
    nontoxic_bonus = ((1 - labels).unsqueeze(0) * (1 - labels).unsqueeze(1)) * 2.0 + 1.0
    weight = same_label * nontoxic_bonus

    # 上三角避免重复
    triu_mask = torch.triu(torch.ones(n, n, device=logits.device), diagonal=1)
    weight = weight * triu_mask

    if weight.sum() < 1e-8:
        return torch.tensor(0.0, device=logits_tox.device, requires_grad=True)

    diffs_sq = (logits.unsqueeze(0) - logits.unsqueeze(1)) ** 2
    loss = (diffs_sq * weight).sum() / (weight.sum() + 1e-8)
    return loss

# =============================================================================
# Evaluation: 计算 Jigsaw Final Metric
# =============================================================================
def power_mean(values, p=-5):
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0: return np.nan
    arr = np.clip(arr, 1e-10, None)
    return float(np.power(np.mean(np.power(arr, p)), 1.0 / p))


def evaluate_with_final_metric(model, loader, device, val_df):
    """评估: 计算 Overall AUC + BiasScore + Final Metric"""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval Fair]"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                out = model(ids, mask)
            probs = torch.sigmoid(out['logits_tox']).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(batch['y_tox'].numpy().flatten())

    probs_arr = np.array(all_probs)
    labels_binary = (np.array(all_labels) >= 0.5).astype(int)

    try:
        overall_auc = roc_auc_score(labels_binary, probs_arr)
    except ValueError:
        overall_auc = 0.5

    # 计算 per-subgroup AUC
    eval_df = val_df.copy()
    eval_df = eval_df.iloc[:len(probs_arr)]  # 对齐长度
    eval_df['_prob'] = probs_arr

    pm_aucs = {'subgroup': [], 'bpsn': [], 'bnsp': []}

    for col in IDENTITY_COLS:
        sub_mask = eval_df[col] >= 0.5
        if sub_mask.sum() == 0:
            continue
        target = (eval_df['y_tox_soft'] if 'y_tox_soft' in eval_df.columns else eval_df['y_tox']) >= 0.5

        # Subgroup AUC
        sub_df = eval_df[sub_mask]
        sub_target = target[sub_mask]
        if sub_target.nunique() >= 2:
            pm_aucs['subgroup'].append(roc_auc_score(sub_target, sub_df['_prob']))

        # BPSN: (subgroup & non-toxic) ∪ (background & toxic)
        bpsn_mask = (sub_mask & ~target) | (~sub_mask & target)
        bpsn_df = eval_df[bpsn_mask]
        bpsn_target = target[bpsn_mask]
        if bpsn_target.nunique() >= 2:
            pm_aucs['bpsn'].append(roc_auc_score(bpsn_target, bpsn_df['_prob']))

        # BNSP: (subgroup & toxic) ∪ (background & non-toxic)
        bnsp_mask = (sub_mask & target) | (~sub_mask & ~target)
        bnsp_df = eval_df[bnsp_mask]
        bnsp_target = target[bnsp_mask]
        if bnsp_target.nunique() >= 2:
            pm_aucs['bnsp'].append(roc_auc_score(bnsp_target, bnsp_df['_prob']))

    pm_sub = power_mean(pm_aucs['subgroup']) if pm_aucs['subgroup'] else 0.5
    pm_bpsn = power_mean(pm_aucs['bpsn']) if pm_aucs['bpsn'] else 0.5
    pm_bnsp = power_mean(pm_aucs['bnsp']) if pm_aucs['bnsp'] else 0.5
    bias_score = (pm_sub + pm_bpsn + pm_bnsp) / 3.0
    final_metric = 0.25 * overall_auc + 0.75 * bias_score

    return {
        'final': final_metric, 'bias_score': bias_score,
        'overall_auc': overall_auc,
        'pm_sub': pm_sub, 'pm_bpsn': pm_bpsn, 'pm_bnsp': pm_bnsp,
    }

# =============================================================================
# Layer-wise LR Decay (复用)
# =============================================================================
def build_layer_wise_param_groups(model, base_lr, decay_factor=0.95):
    param_groups = []
    base_model = model.module if hasattr(model, 'module') else model

    embed_params = list(base_model.deberta.embeddings.parameters())
    if embed_params:
        param_groups.append({"params": embed_params, "lr": base_lr * decay_factor ** 2})

    encoder_layers = base_model.deberta.encoder.layer
    n_layers = len(encoder_layers)
    mid_point = n_layers // 2
    for i, layer in enumerate(encoder_layers):
        lr = base_lr * decay_factor ** 2 if i < mid_point else base_lr * decay_factor
        param_groups.append({"params": list(layer.parameters()), "lr": lr})

    head_params = [p for n, p in base_model.named_parameters()
                   if any(k in n for k in ['proj_tox', 'tox_head', 'att_pooling'])]
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    return param_groups

# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fair DeBERTa-V3 Stage 2 (Slice Ranking + Anchor-PCGrad + CLP)")
    parser.add_argument("--s1_checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--layer_decay", type=float, default=0.95)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--no_bar", action="store_true")
    parser.add_argument("--ablation_tag", type=str, default=None)
    # === 公平性超参 ===
    parser.add_argument("--lambda_bias", type=float, default=1.0, help="公平损失权重上限 λ_max")
    parser.add_argument("--lambda_clp", type=float, default=0.1, help="CLP 损失权重 (0=关闭)")
    parser.add_argument("--debias_start", type=float, default=0.2, help="Debias 阶段起始比例")
    parser.add_argument("--debias_end", type=float, default=0.9, help="Debias 阶段结束比例")
    parser.add_argument("--power_p", type=int, default=4, help="Slice loss power-mean 的 p 值")
    parser.add_argument("--max_pairs", type=int, default=256, help="每种 pair 类型的最大采样数")
    # === Identity-aware Reweighting ===
    parser.add_argument("--w_id_toxic", type=float, default=1.5, help="has_identity & toxic 样本的权重")
    # === 消融 ===
    parser.add_argument("--no_pcgrad", action="store_true", help="消融: 关闭 Anchor-PCGrad")
    parser.add_argument("--no_slice", action="store_true", help="消融: 关闭 Slice Ranking Loss")
    parser.add_argument("--no_clp", action="store_true", help="消融: 关闭 CLP")
    parser.add_argument("--no_reweight", action="store_true", help="消融: 关闭 identity-aware 重加权")

    args = parser.parse_args()

    # --- DDP ---
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        is_main = (args.local_rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    if args.no_bar or not is_main:
        global tqdm
        from tqdm import tqdm as _tqdm
        tqdm = lambda *a, **kw: _tqdm(*a, **kw, disable=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    suffix = '_Fair'
    if args.no_pcgrad: suffix += '_NoPCGrad'
    if args.no_slice: suffix += '_NoSlice'
    if args.no_clp: suffix += '_NoCLP'
    if args.no_reweight: suffix += '_NoReweight'
    if args.ablation_tag: suffix += f'_{args.ablation_tag}'
    save_name = f"DebertaV3Fair_S2{suffix}_Seed{args.seed}_Sample{args.sample_size}_{datetime.now().strftime('%m%d_%H%M')}"
    save_path = get_model_path(save_name + ".pth")

    # --- 数据 ---
    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.data_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

    # --- 模型 ---
    model = DebertaV3MTL(args.model_name).to(device)
    if os.path.exists(args.s1_checkpoint):
        sd = torch.load(args.s1_checkpoint, map_location=device)
        if 'proj_aux.weight' in sd and 'proj_sub.weight' not in sd:
            sd['proj_sub.weight'] = sd['proj_aux.weight'].clone()
            sd['proj_sub.bias'] = sd['proj_aux.bias'].clone()
            sd['proj_id.weight'] = sd['proj_aux.weight'].clone()
            sd['proj_id.bias'] = sd['proj_aux.bias'].clone()
        model.load_state_dict(sd, strict=False)
        if is_main: print(f"  [Success] 加载 S1 权重: {args.s1_checkpoint}")

    # 冻结辅助头: 不再让 identity/subtype 头的梯度更新 backbone
    for n, p in model.named_parameters():
        if any(k in n for k in ['proj_sub', 'proj_id', 'subtype_head', 'identity_head',
                                 'drop_sub', 'drop_id', 'log_var']):
            p.requires_grad_(False)
    if is_main:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  [Freeze] 可训练: {n_train:,} | 冻结(辅助头+log_var): {n_frozen:,}")

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # --- 优化器 ---
    param_groups = build_layer_wise_param_groups(model, base_lr=args.lr, decay_factor=args.layer_decay)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    world_size = dist.get_world_size()
    num_steps = int(len(train_ds) / args.batch_size / world_size / args.grad_accum * args.epochs)
    warmup_steps = int(args.warmup_ratio * num_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    # --- 损失模块 ---
    criterion_main = nn.BCEWithLogitsLoss()  # fallback for no_reweight
    gw_tensor = None if args.no_reweight else get_group_weights_tensor(device)
    slice_loss_fn = SliceRankingLoss(power_p=args.power_p, max_pairs_per_type=args.max_pairs)

    # --- EMA ---
    base_model = model.module if hasattr(model, 'module') else model
    ema = ModelEMA(base_model, decay=args.ema_decay) if args.ema_decay > 0 else None

    best_final = 0.0
    loss_history = {"train": [], "val_final": [], "val_auc": [], "conflict_rate": []}
    global_step = 0

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Fair S2 Training Config:")
        print(f"  λ_bias={args.lambda_bias} | λ_clp={args.lambda_clp} | power_p={args.power_p}")
        print(f"  Schedule: warmup[0-{args.debias_start}] → debias[{args.debias_start}-{args.debias_end}] → stabilize[{args.debias_end}-1.0]")
        print(f"  PCGrad={'ON' if not args.no_pcgrad else 'OFF'} | Slice={'ON' if not args.no_slice else 'OFF'} | CLP={'ON' if not args.no_clp else 'OFF'} | Reweight={'ON' if not args.no_reweight else 'OFF'}")
        print(f"  Total steps: {num_steps} | Warmup: {warmup_steps}")
        print(f"{'='*60}\n")

    # =================================================================
    # 训练循环
    # =================================================================
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss, n_conflicts, n_steps = 0.0, 0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[Fair S2] Epoch {epoch+1}/{args.epochs}")

        for i, batch in enumerate(pbar):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            y_id = batch['y_id'].to(device)
            has_id = batch['has_id'].to(device)

            # --- 三段 Schedule: 动态 λ ---
            progress = global_step / max(num_steps, 1)
            if progress < args.debias_start:
                cur_lambda = 0.0
                cur_clp_lambda = 0.0
            elif progress < args.debias_end:
                ramp = (progress - args.debias_start) / (args.debias_end - args.debias_start)
                cur_lambda = args.lambda_bias * ramp
                cur_clp_lambda = args.lambda_clp * ramp
            else:
                cur_lambda = args.lambda_bias * 0.5
                cur_clp_lambda = args.lambda_clp * 0.5

            with torch.cuda.amp.autocast():
                out = model(ids, mask)
                l_main = weighted_toxicity_loss(
                    out['logits_tox'], y_tox, has_id, y_id,
                    w_id_toxic=args.w_id_toxic,
                    group_weights_tensor=gw_tensor
                ) if not args.no_reweight else criterion_main(out['logits_tox'], y_tox)

                # Slice Ranking Loss
                if not args.no_slice and cur_lambda > 0:
                    l_slice = slice_loss_fn(out['logits_tox'], y_tox, y_id)
                else:
                    l_slice = torch.tensor(0.0, device=device)

                # CLP
                if not args.no_clp and cur_clp_lambda > 0:
                    l_clp = compute_clp_loss(out['logits_tox'], y_tox, y_id, has_id)
                else:
                    l_clp = torch.tensor(0.0, device=device)

            # --- 梯度更新 ---
            l_fair = l_slice + cur_clp_lambda * l_clp if cur_lambda > 0 else None
            use_pcgrad = (not args.no_pcgrad) and (l_fair is not None) and (cur_lambda > 0)

            if use_pcgrad and l_fair.requires_grad:
                conflict = anchor_pcgrad_step(
                    model, l_main, l_fair, cur_lambda,
                    optimizer, scaler, args.grad_accum, ema)
                if conflict: n_conflicts += 1
            else:
                # Warmup 或消融: 只用主任务
                total_fair = l_main
                if l_fair is not None and cur_lambda > 0 and l_fair.requires_grad:
                    total_fair = l_main + cur_lambda * l_fair
                scaled = scaler.scale(total_fair / args.grad_accum)
                scaled.backward()

            n_steps += 1

            if (i + 1) % args.grad_accum == 0:
                if not use_pcgrad or l_fair is None or not l_fair.requires_grad:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # PCGrad 路径已经在 anchor_pcgrad_step 中设好 grad
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if ema is not None and not use_pcgrad:
                    ema.update(base_model)
                if ema is not None and use_pcgrad:
                    ema.update(base_model)

            total_loss += l_main.item()
            pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}", lam=f"{cur_lambda:.3f}",
                             conflict=f"{n_conflicts}/{n_steps}")

        # --- Epoch 评估 ---
        if ema is not None:
            ema.apply_shadow(base_model)

        metrics = evaluate_with_final_metric(model, val_loader, device, val_df)

        dist.barrier()

        if is_main:
            conflict_rate = n_conflicts / max(n_steps, 1)
            loss_history["train"].append(total_loss / len(train_loader))
            loss_history["val_final"].append(metrics['final'])
            loss_history["val_auc"].append(metrics['overall_auc'])
            loss_history["conflict_rate"].append(conflict_rate)

            print(f"  Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}")
            print(f"  [Final={metrics['final']:.4f}] BiasScore={metrics['bias_score']:.4f} | AUC={metrics['overall_auc']:.4f}")
            print(f"  PM(Sub)={metrics['pm_sub']:.4f} | PM(BPSN)={metrics['pm_bpsn']:.4f} | PM(BNSP)={metrics['pm_bnsp']:.4f}")
            print(f"  [PCGrad] Conflict rate: {conflict_rate:.1%}")

            if metrics['final'] > best_final:
                best_final = metrics['final']
                torch.save(base_model.state_dict(), save_path)
                print(f"  [Save] Best Final={metrics['final']:.4f} -> {save_path}")

        if ema is not None:
            ema.restore(base_model)

    # --- 收尾 ---
    if is_main:
        with open(get_log_path(save_name + "_loss.json"), 'w') as f:
            json.dump(loss_history, f, indent=2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ep = range(1, len(loss_history["train"])+1)
        axes[0].plot(ep, loss_history["train"], 'b-o'); axes[0].set_title('Train Loss'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(ep, loss_history["val_final"], 'g-o', label='Final Metric')
        axes[1].plot(ep, loss_history["val_auc"], 'r--o', label='Overall AUC')
        axes[1].set_title('Validation Metrics'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[2].plot(ep, loss_history["conflict_rate"], 'm-o'); axes[2].set_title('PCGrad Conflict Rate'); axes[2].grid(True, alpha=0.3)
        plt.suptitle(f'Fair S2: {save_name}')
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150); plt.close()
        print(f"\n>>> 实验完成。Best Final Metric={best_final:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
