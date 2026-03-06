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
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import copy
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
# ### 核心训练脚本：train_deberta_mtl_stage2.py ###
# =============================================================================

IDENTITY_COLS = [
    'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian',
    'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
]

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_deberta_v3_mtl import DebertaV3MTL
from exp_data_loader import ToxicityDataset, sample_aligned_data
from path_config import get_model_path, get_log_path
from train_utils import EarlyStopping

# =============================================================================
# EMA (Exponential Moving Average) — 平滑模型权重，提升泛化
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
# 顺序与 identity_cols 一致:
#   male, female, black, white, muslim, jewish, christian,
#   homosexual_gay_or_lesbian, psychiatric_or_mental_illness
# =============================================================================
IDENTITY_GROUP_WEIGHTS = [1.2, 1.0, 2.3, 1.8, 1.9, 2.9, 1.3, 2.6, 3.4]

# 预创建为 tensor 避免每步重复分配 (P2 优化)
_GROUP_WEIGHTS_TENSOR = None

def get_group_weights_tensor(device):
    global _GROUP_WEIGHTS_TENSOR
    if _GROUP_WEIGHTS_TENSOR is None or _GROUP_WEIGHTS_TENSOR.device != device:
        _GROUP_WEIGHTS_TENSOR = torch.tensor(IDENTITY_GROUP_WEIGHTS, dtype=torch.float, device=device)
    return _GROUP_WEIGHTS_TENSOR

def weighted_toxicity_loss(logits, targets, has_id, y_id, w_id_toxic=1.5, group_weights_tensor=None):
    """
    身份感知加权损失函数 (per-group differential weighting)
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

def uncertainty_weighted_loss(l_tox, l_sub, l_id, log_var_tox, log_var_sub, log_var_id):
    """Uncertainty Weighting (Kendall et al., 2018)"""
    w_tox = l_tox / (2 * torch.exp(log_var_tox)) + log_var_tox / 2
    w_sub = l_sub / (2 * torch.exp(log_var_sub)) + log_var_sub / 2
    w_id = l_id / (2 * torch.exp(log_var_id)) + log_var_id / 2
    return w_tox + w_sub + w_id

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps, w_id_toxic, no_reweight=False, only_toxicity=False, aux_scale=None, ema=None):
    model.train()
    criterion_aux = nn.BCEWithLogitsLoss()
    gw_tensor = None if no_reweight else get_group_weights_tensor(device)
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="[Train S2 AMP]")
    for i, batch in enumerate(pbar):
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        y_tox, y_sub, y_id, has_id = batch['y_tox'].to(device).unsqueeze(-1), batch['y_sub'].to(device), batch['y_id'].to(device), batch['has_id'].to(device)

        with torch.cuda.amp.autocast():
            out = model(ids, mask)
            if no_reweight:
                l_tox = criterion_aux(out['logits_tox'], y_tox)
            else:
                l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, y_id,
                                               w_id_toxic=w_id_toxic,
                                               group_weights_tensor=gw_tensor)
            if only_toxicity:
                loss = l_tox
            elif aux_scale is not None:
                # 固定权重: toxicity 占主导, 辅助任务按 aux_scale 缩放
                l_sub = criterion_aux(out['logits_sub'], y_sub)
                l_id = criterion_aux(out['logits_id'], y_id)
                loss = l_tox + aux_scale * (l_sub + l_id)
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
            # EMA update after each optimizer step
            if ema is not None:
                ema.update(model.module if hasattr(model, 'module') else model)
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")
    return total_loss / len(loader)

def evaluate(model, loader, device, w_id_toxic=1.5, no_reweight=False, aux_scale=None):
    """评估: 计算 loss + AUC"""
    model.eval()
    criterion_aux = nn.BCEWithLogitsLoss()
    gw_tensor = None if no_reweight else get_group_weights_tensor(device)
    total_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval S2]"):
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            has_id = batch['has_id'].to(device)
            y_id = batch['y_id'].to(device)
            y_sub = batch['y_sub'].to(device)

            with torch.cuda.amp.autocast():
                out = model(ids, mask)
                if no_reweight:
                    l_tox = criterion_aux(out['logits_tox'], y_tox)
                else:
                    l_tox = weighted_toxicity_loss(out['logits_tox'], y_tox, has_id, y_id,
                                                  w_id_toxic=w_id_toxic,
                                                  group_weights_tensor=gw_tensor)
                l_sub = criterion_aux(out['logits_sub'], y_sub)
                l_id = criterion_aux(out['logits_id'], y_id)
                if aux_scale is not None:
                    loss = l_tox + aux_scale * (l_sub + l_id)
                else:
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
        for batch in tqdm(loader, desc="[Eval Final]"):
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
    eval_df = val_df.copy().iloc[:len(probs_arr)]
    eval_df['_prob'] = probs_arr
    pm_aucs = {'subgroup': [], 'bpsn': [], 'bnsp': []}
    for col in IDENTITY_COLS:
        sub_mask = eval_df[col] >= 0.5
        if sub_mask.sum() == 0: continue
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

def build_layer_wise_param_groups(model, base_lr, decay_factor=0.8):
    """
    Layer-wise learning rate decay (P1: 防止灾难遗忘)
    底层 (embeddings + 前6层): lr * decay^2 = 0.64x
    中层 (后6层):              lr * decay   = 0.8x
    顶层 (heads + projection): lr * 1.0x
    """
    param_groups = []

    # DDP wrapping: 通过 model.module 访问实际模型
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
            lr = base_lr * decay_factor ** 2  # 底层: 0.64x
        else:
            lr = base_lr * decay_factor        # 上层: 0.8x
        param_groups.append({"params": layer_params, "lr": lr})

    # 3. Heads + Projection + log_var: 最大 lr
    head_params = []
    for name, param in base_model.named_parameters():
        if any(k in name for k in ['proj_tox', 'proj_sub', 'proj_id', 'proj_aux', 'tox_head', 'subtype_head', 'identity_head', 'att_pooling', 'log_var', 'drop_']):
            head_params.append(param)
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    return param_groups

def main():
    parser = argparse.ArgumentParser(description="DeBERTa-V3 Stage 2 MTL (AMP)")
    parser.add_argument("--s1_checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--scheduler", type=str, choices=["linear", "cosine", "plateau"], default="linear")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--w_identity", type=float, default=2.5)
    parser.add_argument("--w_id_toxic", type=float, default=1.5)
    parser.add_argument("--no_reweight", action="store_true")
    parser.add_argument("--no_bar", action="store_true")

    # Ablation Flags
    parser.add_argument("--no_pooling", action="store_true")
    parser.add_argument("--only_toxicity", action="store_true")
    parser.add_argument("--no_focal", action="store_true")
    parser.add_argument("--ablation_tag", type=str, default=None)
    parser.add_argument("--layer_decay", type=float, default=0.8, help="Layer-wise LR decay factor, 1.0=no decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--aux_scale", type=float, default=None, help="Fixed aux loss scale (替代uncertainty weighting). 例如0.3=辅助任务权重0.3")
    parser.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay rate. 0=不用EMA, 0.999=常用值")
    parser.add_argument("--select_by_final", action="store_true", help="用 Final Metric 选 checkpoint (而非 AUC)")

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

    # Suffix Construction
    suffix = ''
    if args.no_reweight: suffix += '_NoReweight'
    if args.no_pooling: suffix += '_NoPooling'
    if args.only_toxicity: suffix += '_OnlyTox'
    if args.no_focal: suffix += '_NoFocal'
    if args.ablation_tag: suffix += f"_{args.ablation_tag}"

    save_name = f"DebertaV3MTL_S2{suffix}_Seed{args.seed}_Sample{args.sample_size}_{datetime.now().strftime('%m%d_%H%M')}"
    save_path = get_model_path(save_name + ".pth")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.data_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ToxicityDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=args.max_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
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
    if os.path.exists(args.s1_checkpoint):
        state_dict = torch.load(args.s1_checkpoint, map_location=device)
        # 热启动: 如果 S1 有 proj_aux 但新模型用 proj_sub/proj_id，则复制权重
        if 'proj_aux.weight' in state_dict and 'proj_sub.weight' not in state_dict:
            state_dict['proj_sub.weight'] = state_dict['proj_aux.weight'].clone()
            state_dict['proj_sub.bias'] = state_dict['proj_aux.bias'].clone()
            state_dict['proj_id.weight'] = state_dict['proj_aux.weight'].clone()
            state_dict['proj_id.bias'] = state_dict['proj_aux.bias'].clone()
            if is_main_process:
                print(f"  [Warmstart] proj_aux → proj_sub + proj_id 权重复制完成")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main_process:
            if missing:
                print(f"  [Info] 新增参数 (随机初始化): {missing}")
            print(f"  [Success] 加载 S1 权重成功。")

    # --- DDP Model Wrapping (删除无用的 SyncBatchNorm) ---
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Layer-wise lr decay (P1: 防止灾难遗忘)
    param_groups = build_layer_wise_param_groups(model, base_lr=args.lr, decay_factor=getattr(args, 'layer_decay', 0.8))
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
    else:
        world_size = dist.get_world_size()
        num_steps = int(len(train_ds) / args.batch_size / world_size / args.grad_accum * args.epochs)
        warmup_steps = int(args.warmup_ratio * num_steps)
        if args.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    early_stopping, best_val_auc = EarlyStopping(patience=args.early_patience), 0.0
    best_final = 0.0
    loss_history = {"train": [], "val": [], "val_auc": [], "val_final": []}

    # 初始化 EMA
    base_model = model.module if hasattr(model, 'module') else model
    ema = ModelEMA(base_model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # 当使用 aux_scale 时，冻结无用的 log_var 参数
    if args.aux_scale is not None:
        for p in [base_model.log_var_tox, base_model.log_var_sub, base_model.log_var_id]:
            p.requires_grad_(False)
        if is_main_process:
            print(f"  [Config] 使用固定 aux_scale={args.aux_scale}, 已冻结 log_var 参数")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if is_main_process: print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device,
                                     args.grad_accum, args.w_id_toxic, args.no_reweight, args.only_toxicity,
                                     aux_scale=args.aux_scale, ema=ema)

        # 评估: 如果有 EMA，用 EMA 权重评估
        if ema is not None:
            ema.apply_shadow(base_model)
        val_loss, val_auc = evaluate(model, val_loader, device, args.w_id_toxic, args.no_reweight, aux_scale=args.aux_scale)

        # Final Metric 评估 (如果启用)
        final_metrics = None
        if args.select_by_final:
            final_metrics = evaluate_with_final_metric(model, val_loader, device, val_df)

        dist.barrier()

        if is_main_process:
            if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(val_loss)
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            loss_history["val_auc"].append(val_auc)
            loss_history["val_final"].append(final_metrics['final'] if final_metrics else 0.0)

            print(f"  Result -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            if final_metrics:
                print(f"  [Final={final_metrics['final']:.4f}] BiasScore={final_metrics['bias_score']:.4f}")
                print(f"  PM(Sub)={final_metrics['pm_sub']:.4f} | PM(BPSN)={final_metrics['pm_bpsn']:.4f} | PM(BNSP)={final_metrics['pm_bnsp']:.4f}")
            if ema is not None:
                print(f"  [EMA] decay={args.ema_decay}")

            # Checkpoint 选择
            if args.select_by_final and final_metrics:
                if final_metrics['final'] > best_final:
                    best_final = final_metrics['final']
                    torch.save(base_model.state_dict(), save_path)
                    print(f"  [Save] Best Final={final_metrics['final']:.4f} -> {save_path}")
            else:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(base_model.state_dict(), save_path)
                    print(f"  [Save] Best AUC={val_auc:.4f} -> {save_path}")

            if early_stopping(val_loss):
                print(f">>> [Early Stop] 提前结束。Best AUC={best_val_auc:.4f}")

        # 恢复非EMA权重继续训练
        if ema is not None:
            ema.restore(base_model)

        stop_signal = torch.tensor(1 if (is_main_process and early_stopping.early_stop) else 0).to(device)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            break

    if is_main_process:
        with open(get_log_path(save_name + "_loss.json"), 'w') as f: json.dump(loss_history, f, indent=2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(loss_history["train"])+1)
        ax1.plot(epochs_range, loss_history["train"], 'b-o', label='Train Loss')
        ax1.plot(epochs_range, loss_history["val"], 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch'); ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_range, loss_history["val_auc"], 'g-o', label='Val AUC')
        ax2.set_xlabel('Epoch'); ax2.set_title('Validation AUC'); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.suptitle(f'S2: {save_name}')
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150); plt.close()
        print(f">>> 实验完成。Best AUC={best_val_auc:.4f}" + (f" | Best Final={best_final:.4f}" if args.select_by_final else ""))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
