"""
=============================================================================
统一评估脚本: 支持所有 Baseline + 我们的方法
7 个指标:
  A. 任务性能: Macro-F1, AUC-ROC
  B. 因果公平: CFR (翻转率), CTFG (预测差异)
  C. 群体公平: FPED (假阳性差异), FNED (假阴性差异)
  D. Per-group F1 Std (群体间 F1 标准差)

支持模型:
  - vanilla: DebertaV3Vanilla
  - ccdf: DebertaV3CCDF + BiasOnlyModel (TDE推理)
  - getfair: DebertaV3CausalFair (梯度公平)
  - ear: DebertaV3CausalFair (注意力熵)
  - davani: DebertaV3Davani (Logit Pairing)
  - ramponi: DebertaV3Ramponi (对抗去偏)
  - ours: DebertaV3CausalFair (因果公平)
=============================================================================
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "train"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from path_config import get_eval_path


# =============================================================================
# 模型加载器: 根据 method 加载对应模型
# =============================================================================
def load_model(method, checkpoint, model_name, device):
    """
    加载模型，返回 (model, predict_fn)
    predict_fn(model, input_ids, attention_mask) -> logits
    """
    if method == "vanilla":
        from train_baseline_vanilla import DebertaV3Vanilla
        model = DebertaV3Vanilla(model_name, num_classes=2).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

        def predict_fn(m, ids, mask):
            return m(ids, mask)['logits']
        return model, predict_fn

    elif method == "ccdf":
        from train_baseline_ccdf import DebertaV3CCDF, BiasOnlyModel, build_identity_mask
        tokenizer_for_mask = AutoTokenizer.from_pretrained(model_name)
        ckpt = torch.load(checkpoint, map_location=device)
        model = DebertaV3CCDF(model_name, num_classes=2).to(device)
        model.load_state_dict(ckpt['main_model'])
        vocab_size = ckpt.get('vocab_size', tokenizer_for_mask.vocab_size)
        bias_model = BiasOnlyModel(vocab_size, embed_dim=128, num_classes=2).to(device)
        bias_model.load_state_dict(ckpt['bias_model'])
        bias_model.eval()
        tde_alpha = ckpt.get('tde_alpha', 0.5)

        def predict_fn(m, ids, mask):
            main_logits = m(ids, mask)['logits']
            id_mask = build_identity_mask(ids, tokenizer_for_mask).to(device)
            bias_logits = bias_model(ids, id_mask)
            return main_logits - tde_alpha * bias_logits
        return model, predict_fn

    elif method in ("getfair", "ear"):
        # EAR/GetFair 用的是 Vanilla 架构 (无 projector)
        from train_baseline_vanilla import DebertaV3Vanilla
        model = DebertaV3Vanilla(model_name, num_classes=2).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

        def predict_fn(m, ids, mask):
            return m(ids, mask)['logits']
        return model, predict_fn

    elif method == "ours":
        from model_deberta_cf import DebertaV3CausalFair
        model = DebertaV3CausalFair(model_name, num_classes=2).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

        def predict_fn(m, ids, mask):
            return m(ids, mask, return_features=False)['logits']
        return model, predict_fn

    elif method == "davani":
        from train_baseline_davani import DebertaV3Davani
        model = DebertaV3Davani(model_name, num_classes=2).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

        def predict_fn(m, ids, mask):
            return m(ids, mask)['logits']
        return model, predict_fn

    elif method == "ramponi":
        from train_baseline_ramponi import DebertaV3Ramponi
        model = DebertaV3Ramponi(model_name, num_classes=2).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

        def predict_fn(m, ids, mask):
            return m(ids, mask, alpha=0.0)['logits']  # 推理时不需要GRL
        return model, predict_fn

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 批量预测
# =============================================================================
@torch.no_grad()
def predict_batch(model, predict_fn, tokenizer, texts, device,
                  max_len=128, batch_size=32):
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, max_length=max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)
        with torch.cuda.amp.autocast():
            logits = predict_fn(model, ids, mask)
            probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


# =============================================================================
# 公平性指标
# =============================================================================
def counterfactual_flip_rate(preds_orig, preds_cf):
    """CFR: 反事实翻转率 (理想: 0.0)"""
    flips = (preds_orig != preds_cf).sum()
    return float(flips) / len(preds_orig) if len(preds_orig) > 0 else 0.0


def counterfactual_token_fairness_gap(probs_orig, probs_cf):
    """CTFG: 反事实预测差异 (理想: 0.0)"""
    return float(np.mean(np.abs(probs_orig - probs_cf))) if len(probs_orig) > 0 else 0.0


def compute_fped_fned(y_true, y_pred, groups):
    """FPED/FNED: 假阳性/假阴性率均等差异 (理想: 0.0)"""
    unique_groups = sorted(set(groups))

    def fpr(yt, yp):
        tn_fp = (yt == 0).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        return float(fp) / max(float(tn_fp), 1)

    def fnr(yt, yp):
        tp_fn = (yt == 1).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        return float(fn) / max(float(tp_fn), 1)

    overall_fpr = fpr(y_true, y_pred)
    overall_fnr = fnr(y_true, y_pred)
    fped, fned = 0.0, 0.0
    group_details = {}

    for g in unique_groups:
        mask = np.array(groups) == g
        if mask.sum() < 5:
            continue
        g_fpr = fpr(y_true[mask], y_pred[mask])
        g_fnr = fnr(y_true[mask], y_pred[mask])
        fped += abs(overall_fpr - g_fpr)
        fned += abs(overall_fnr - g_fnr)
        group_details[str(g)] = {
            'fpr': round(g_fpr, 4), 'fnr': round(g_fnr, 4),
            'fpr_gap': round(abs(overall_fpr - g_fpr), 4),
            'fnr_gap': round(abs(overall_fnr - g_fnr), 4),
            'count': int(mask.sum()),
        }

    return fped, fned, group_details


# =============================================================================
# 主评估流程
# =============================================================================
def evaluate_all(model, predict_fn, tokenizer, test_df, cf_test_df,
                 device, max_len=128, threshold=0.5):
    results = {}
    texts = test_df['text'].tolist()
    labels = test_df['binary_label'].values

    # ---- A. 任务性能 ----
    print("\n[A] 任务性能...")
    probs = predict_batch(model, predict_fn, tokenizer, texts, device, max_len)
    preds = (probs >= threshold).astype(int)

    results['macro_f1'] = round(float(f1_score(labels, preds, average='macro')), 4)
    results['auc_roc'] = round(float(roc_auc_score(labels, probs)), 4) if len(set(labels)) > 1 else 0.5
    results['accuracy'] = round(float(accuracy_score(labels, preds)), 4)
    results['binary_f1'] = round(float(f1_score(labels, preds, average='binary')), 4)
    print(f"  Macro-F1={results['macro_f1']}  AUC={results['auc_roc']}")

    # ---- B. 因果公平 ----
    results['cfr'] = None
    results['ctfg'] = None
    if cf_test_df is not None and len(cf_test_df) > 0:
        print("[B] 因果公平...")
        cf_originals = cf_test_df['original_text'].tolist()
        cf_texts = cf_test_df['cf_text'].tolist()

        probs_orig = predict_batch(model, predict_fn, tokenizer, cf_originals, device, max_len)
        probs_cf = predict_batch(model, predict_fn, tokenizer, cf_texts, device, max_len)
        preds_orig = (probs_orig >= threshold).astype(int)
        preds_cf = (probs_cf >= threshold).astype(int)

        results['cfr'] = round(counterfactual_flip_rate(preds_orig, preds_cf), 4)
        results['ctfg'] = round(counterfactual_token_fairness_gap(probs_orig, probs_cf), 4)
        print(f"  CFR={results['cfr']}  CTFG={results['ctfg']}")

    # ---- C. 群体公平 ----
    results['fped'] = None
    results['fned'] = None
    results['per_group_f1_std'] = None

    group_col = None
    if 'target_group' in test_df.columns:
        group_col = 'target_group'
    elif 'coarse_groups' in test_df.columns:
        test_df = test_df.copy()
        test_df['_primary_group'] = test_df['coarse_groups'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'none'
        )
        group_col = '_primary_group'

    if group_col and test_df[group_col].nunique() > 1:
        print("[C] 群体公平...")
        groups = test_df[group_col].values
        fped, fned, group_details = compute_fped_fned(labels, preds, groups)
        results['fped'] = round(fped, 4)
        results['fned'] = round(fned, 4)
        results['group_details'] = group_details
        print(f"  FPED={results['fped']}  FNED={results['fned']}")

        # Per-group F1 标准差
        per_group_f1 = {}
        for g in sorted(set(groups)):
            mask = np.array(groups) == g
            if mask.sum() >= 5 and len(set(labels[mask])) > 1:
                per_group_f1[str(g)] = float(f1_score(labels[mask], preds[mask], average='macro'))
        if per_group_f1:
            f1_values = list(per_group_f1.values())
            results['per_group_f1_std'] = round(float(np.std(f1_values)), 4)
            results['per_group_f1'] = {k: round(v, 4) for k, v in per_group_f1.items()}
            print(f"  Per-group F1 Std={results['per_group_f1_std']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="统一 Baseline 评估 (7指标)")
    parser.add_argument("--method", type=str, required=True,
                        choices=["vanilla", "ccdf", "getfair", "ear", "davani", "ramponi", "ours"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen", "dynahate"])
    parser.add_argument("--cf_method", type=str, default="llm",
                        choices=["swap", "llm"])
    parser.add_argument("--model_name", type=str,
                        default=os.path.join(BASE_DIR, "models", "deberta-v3-base"))
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径 (默认自动生成)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] method={args.method} dataset={args.dataset} cf={args.cf_method}")
    print(f"[Config] checkpoint={args.checkpoint}")

    # 加载模型
    model, predict_fn = load_model(args.method, args.checkpoint, args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 加载数据
    test_df = pd.read_parquet(os.path.join(args.data_dir, f"{args.dataset}_test.parquet"))
    cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
    cf_path = os.path.join(args.data_dir, f"{args.dataset}_test_{cf_suffix}.parquet")
    cf_test_df = pd.read_parquet(cf_path) if os.path.exists(cf_path) else None
    if cf_test_df is not None:
        print(f"[CF] {len(cf_test_df)} 条反事实 ({args.cf_method})")

    # 评估
    results = evaluate_all(model, predict_fn, tokenizer, test_df, cf_test_df,
                           device, args.max_len, args.threshold)
    results['_meta'] = {
        'method': args.method, 'dataset': args.dataset,
        'cf_method': args.cf_method, 'checkpoint': args.checkpoint,
    }

    # 保存
    if args.output:
        save_path = args.output
    else:
        ckpt_name = os.path.basename(args.checkpoint).replace('.pth', '')
        save_path = get_eval_path(f"{ckpt_name}_7metrics_{args.cf_method}.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  {args.method.upper()} on {args.dataset} ({args.cf_method})")
    print(f"{'='*60}")
    print(f"  Macro-F1:  {results['macro_f1']}")
    print(f"  AUC-ROC:   {results['auc_roc']}")
    print(f"  CFR:       {results['cfr']}")
    print(f"  CTFG:      {results['ctfg']}")
    print(f"  FPED:      {results['fped']}")
    print(f"  FNED:      {results['fned']}")
    print(f"  F1-Std:    {results['per_group_f1_std']}")
    print(f"{'='*60}")
    print(f"保存至: {save_path}")


if __name__ == "__main__":
    main()
