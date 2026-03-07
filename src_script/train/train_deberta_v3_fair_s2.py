"""
=============================================================================
### 公平性训练脚本: train_deberta_v3_fair_s2.py ###
论文核心方法实现 — 在原有 MTL S2 基础上加入公平性约束:

核心思路:
  保持多任务头全部活跃 (toxicity + subtype + identity),
  利用辅助任务对 backbone 的正则化来维持身份感知表征能力.
  在此基础上添加 CLP (Counterfactual Logit Pairing) 消除身份词捷径,
  并用 Jigsaw Official Final Metric 做 checkpoint 选择.

关键组件:
  A. MTL: 3 任务头全部活跃, 辅助任务以 aux_scale 加权
  B. Identity-aware Reweighting: 按身份稀有度差异化加权
  C. CLP: 反事实 Logit Pairing, 消除身份词捷径 (BPSN 场景)
  D. Final Metric: 0.25*Overall_AUC + 0.75*BiasScore 做 checkpoint 选择
  E. Cosine Schedule + EMA + Layer-wise LR Decay
=============================================================================
"""
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
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
# 身份分组差异化权重 (V5: 温和定向加权)
# 顺序与 IDENTITY_COLS 一致:
#   male, female, black, white, muslim, jewish, christian,
#   homosexual_gay_or_lesbian, psychiatric_or_mental_illness
# V5: 针对 per-subgroup 短板做温和提升
#   homosexual 2.6→3.5, black 2.3→3.0, white 1.8→2.5 (其余不变)
# =============================================================================
IDENTITY_GROUP_WEIGHTS = [1.2, 1.0, 3.0, 2.5, 1.9, 2.9, 1.3, 3.5, 3.4]
_GROUP_WEIGHTS_TENSOR = None

def get_group_weights_tensor(device):
    global _GROUP_WEIGHTS_TENSOR
    if _GROUP_WEIGHTS_TENSOR is None or _GROUP_WEIGHTS_TENSOR.device != device:
        _GROUP_WEIGHTS_TENSOR = torch.tensor(IDENTITY_GROUP_WEIGHTS, dtype=torch.float, device=device)
    return _GROUP_WEIGHTS_TENSOR

def weighted_toxicity_loss(logits, targets, has_id, y_id, w_id_toxic=1.5, group_weights_tensor=None):
    """
    身份感知加权损失:
    - non-toxic + identity: 按身份稀有度差异化加权 (防止假阳性)
    - toxic + identity: 统一 w_id_toxic 权重
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
# EMA (Exponential Moving Average)
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
# CLP (Counterfactual Logit Pairing) 近似
#   约束同 toxicity-label 的不同 identity 样本 logit 一致
#   消除身份词捷径, 改善 BPSN (误报) 场景
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

    eval_df = val_df.copy()
    eval_df = eval_df.iloc[:len(probs_arr)]
    eval_df['_prob'] = probs_arr

    pm_aucs = {'subgroup': [], 'bpsn': [], 'bnsp': []}

    for col in IDENTITY_COLS:
        sub_mask = eval_df[col] >= 0.5
        if sub_mask.sum() == 0:
            continue
        target = (eval_df['y_tox_soft'] if 'y_tox_soft' in eval_df.columns else eval_df['y_tox']) >= 0.5

        sub_df = eval_df[sub_mask]
        sub_target = target[sub_mask]
        if sub_target.nunique() >= 2:
            pm_aucs['subgroup'].append(roc_auc_score(sub_target, sub_df['_prob']))

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
        'final': final_metric, 'bias_score': bias_score,
        'overall_auc': overall_auc,
        'pm_sub': pm_sub, 'pm_bpsn': pm_bpsn, 'pm_bnsp': pm_bnsp,
    }

# =============================================================================
# Layer-wise LR Decay
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

    # 所有 head 参数 (toxicity + subtype + identity + attention pooling)
    head_params = []
    for n, p in base_model.named_parameters():
        if any(k in n for k in ['projection',
                                 'tox_head', 'subtype_head', 'identity_head',
                                 'att_pooling', 'dropout']):
            head_params.append(p)
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    return param_groups

# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fair DeBERTa-V3 Stage 2 (MTL + CLP)")
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
    # === MTL 参数 ===
    parser.add_argument("--aux_scale", type=float, default=0.3, help="辅助任务损失权重 (固定)")
    # === 公平性超参 ===
    parser.add_argument("--lambda_clp", type=float, default=0.1, help="CLP 损失权重")
    parser.add_argument("--clp_warmup", type=float, default=0.2, help="CLP 介入的训练进度阈值")
    # === Identity-aware Reweighting ===
    parser.add_argument("--w_id_toxic", type=float, default=1.5, help="has_identity & toxic 样本权重")
    # === 消融 ===
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
        model.load_state_dict(sd, strict=True)
        if is_main: print(f"  [Success] 加载 S1 权重 (strict=True): {args.s1_checkpoint}")

    if is_main:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [MTL] 可训练参数: {n_train:,}")
        print(f"  [MTL] 辅助任务头全部活跃, aux_scale={args.aux_scale}")

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True)

    # --- 优化器 ---
    param_groups = build_layer_wise_param_groups(model, base_lr=args.lr, decay_factor=args.layer_decay)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    world_size = dist.get_world_size()
    num_steps = int(len(train_ds) / args.batch_size / world_size / args.grad_accum * args.epochs)
    warmup_steps = int(args.warmup_ratio * num_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=num_steps)

    # --- 损失模块 ---
    criterion_aux = nn.BCEWithLogitsLoss()
    criterion_fallback = nn.BCEWithLogitsLoss()
    gw_tensor = None if args.no_reweight else get_group_weights_tensor(device)

    # --- EMA ---
    base_model = model.module if hasattr(model, 'module') else model
    ema = ModelEMA(base_model, decay=args.ema_decay) if args.ema_decay > 0 else None

    best_final = 0.0
    loss_history = {"train": [], "val_final": [], "val_auc": [], "val_bias": []}
    global_step = 0

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Fair S2 V5 Training Config (MTL + CLP):")
        print(f"  aux_scale={args.aux_scale} | λ_clp={args.lambda_clp} | clp_warmup={args.clp_warmup}")
        print(f"  CLP={'ON' if not args.no_clp else 'OFF'} | Reweight={'ON' if not args.no_reweight else 'OFF'}")
        print(f"  Group Weights: {IDENTITY_GROUP_WEIGHTS}")
        print(f"  EMA={args.ema_decay} | Layer Decay={args.layer_decay} | LR={args.lr}")
        print(f"  Total steps: {num_steps} | Warmup: {warmup_steps}")
        print(f"{'='*60}\n")

    # =================================================================
    # 训练循环
    # =================================================================
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[Fair S2] Epoch {epoch+1}/{args.epochs}")

        for i, batch in enumerate(pbar):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device).unsqueeze(-1)
            y_sub = batch['y_sub'].to(device)
            y_id = batch['y_id'].to(device)
            has_id = batch['has_id'].to(device)

            progress = global_step / max(num_steps, 1)

            with torch.cuda.amp.autocast():
                out = model(ids, mask)

                # 主任务: 身份感知加权毒性损失
                if not args.no_reweight:
                    l_tox = weighted_toxicity_loss(
                        out['logits_tox'], y_tox, has_id, y_id,
                        w_id_toxic=args.w_id_toxic,
                        group_weights_tensor=gw_tensor)
                else:
                    l_tox = criterion_fallback(out['logits_tox'], y_tox)

                # 辅助任务: subtype + identity (保持 MTL 正则化)
                l_sub = criterion_aux(out['logits_sub'], y_sub)
                l_id = criterion_aux(out['logits_id'], y_id)

                # CLP: 前 clp_warmup 阶段不用, 之后线性引入
                if not args.no_clp and progress >= args.clp_warmup:
                    l_clp = compute_clp_loss(out['logits_tox'], y_tox, y_id, has_id)
                    clp_ramp = min(1.0, (progress - args.clp_warmup) / 0.2)
                    cur_clp_w = args.lambda_clp * clp_ramp
                else:
                    l_clp = torch.tensor(0.0, device=device)
                    cur_clp_w = 0.0

                # 总损失: MTL + CLP
                loss = l_tox + args.aux_scale * (l_sub + l_id) + cur_clp_w * l_clp

            scaled = scaler.scale(loss / args.grad_accum)
            scaled.backward()

            if (i + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if ema is not None:
                    ema.update(base_model)

            total_loss += l_tox.item()
            pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}",
                             clp=f"{cur_clp_w:.3f}")

        # --- Epoch 评估 ---
        if ema is not None:
            ema.apply_shadow(base_model)

        metrics = evaluate_with_final_metric(model, val_loader, device, val_df)

        dist.barrier()

        if is_main:
            loss_history["train"].append(total_loss / len(train_loader))
            loss_history["val_final"].append(metrics['final'])
            loss_history["val_auc"].append(metrics['overall_auc'])
            loss_history["val_bias"].append(metrics['bias_score'])

            print(f"  Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}")
            print(f"  [Final={metrics['final']:.4f}] BiasScore={metrics['bias_score']:.4f} | AUC={metrics['overall_auc']:.4f}")
            print(f"  PM(Sub)={metrics['pm_sub']:.4f} | PM(BPSN)={metrics['pm_bpsn']:.4f} | PM(BNSP)={metrics['pm_bnsp']:.4f}")

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

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ep = range(1, len(loss_history["train"])+1)
        axes[0].plot(ep, loss_history["train"], 'b-o')
        axes[0].set_title('Train Loss'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(ep, loss_history["val_final"], 'g-o', label='Final Metric')
        axes[1].plot(ep, loss_history["val_auc"], 'r--o', label='Overall AUC')
        axes[1].plot(ep, loss_history["val_bias"], 'b--s', label='BiasScore')
        axes[1].set_title('Validation Metrics'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.suptitle(f'Fair S2 V5: {save_name}')
        plt.savefig(get_log_path(save_name + "_loss.png"), dpi=150); plt.close()
        print(f"\n>>> 实验完成。Best Final Metric={best_final:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
