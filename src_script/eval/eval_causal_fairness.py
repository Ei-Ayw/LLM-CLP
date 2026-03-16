"""
=============================================================================
因果公平评估脚本
评估维度:
  A. 任务性能: Macro-F1, AUC, Per-group F1
  B. 因果公平: CFR (翻转率), CTFG (预测差异)
  C. 群体公平: FPED (假阳性差异), FNED (假阴性差异)
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
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "utils"))

from model_deberta_cf import DebertaV3CausalFair
from path_config import get_eval_path


# =====================================================
# 因果公平指标
# =====================================================
def counterfactual_flip_rate(preds_orig, preds_cf):
    """
    CFR: 反事实翻转率
    CFR = 换了群体后预测标签发生变化的比例
    理想值: 0.0 (完全公平)
    """
    flips = (preds_orig != preds_cf).sum()
    return float(flips) / len(preds_orig) if len(preds_orig) > 0 else 0.0


def counterfactual_token_fairness_gap(probs_orig, probs_cf):
    """
    CTFG: 反事实预测差异
    CTFG = mean(|P(toxic|x) - P(toxic|x_cf)|)
    理想值: 0.0
    """
    gaps = np.abs(probs_orig - probs_cf)
    return float(np.mean(gaps)) if len(gaps) > 0 else 0.0


def compute_fped_fned(y_true, y_pred, groups):
    """
    FPED: 假阳性率均等差异
    FNED: 假阴性率均等差异
    理想值: 0.0
    """
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
        if mask.sum() == 0:
            continue
        g_fpr = fpr(y_true[mask], y_pred[mask])
        g_fnr = fnr(y_true[mask], y_pred[mask])
        fped += abs(overall_fpr - g_fpr)
        fned += abs(overall_fnr - g_fnr)
        group_details[g] = {
            'fpr': g_fpr, 'fnr': g_fnr,
            'fpr_gap': abs(overall_fpr - g_fpr),
            'fnr_gap': abs(overall_fnr - g_fnr),
            'count': int(mask.sum()),
        }

    return fped, fned, group_details


# =====================================================
# 模型推理
# =====================================================
@torch.no_grad()
def predict_batch(model, tokenizer, texts, device, max_len=128, batch_size=32):
    """批量预测，返回概率"""
    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, max_length=max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.cuda.amp.autocast():
            out = model(input_ids, attention_mask, return_features=False)
            probs = F.softmax(out['logits'], dim=-1)[:, 1]  # P(toxic)

        all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


# =====================================================
# 主评估流程
# =====================================================
def evaluate_causal_fairness(model, tokenizer, test_df, cf_test_df,
                              device, max_len=128, threshold=0.5):
    """
    完整的因果公平评估

    Returns:
        dict with all metrics
    """
    results = {}
    texts = test_df['text'].tolist()
    labels = test_df['binary_label'].values

    # ---- A. 任务性能 ----
    print("\n[A] 任务性能评估...")
    probs = predict_batch(model, tokenizer, texts, device, max_len)
    preds = (probs >= threshold).astype(int)

    results['task'] = {
        'accuracy': float(accuracy_score(labels, preds)),
        'macro_f1': float(f1_score(labels, preds, average='macro')),
        'binary_f1': float(f1_score(labels, preds, average='binary')),
        'auc_roc': float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.5,
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
    }
    print(f"  Macro-F1: {results['task']['macro_f1']:.4f}")
    print(f"  AUC-ROC:  {results['task']['auc_roc']:.4f}")

    # ---- B. 因果公平 (需要反事实) ----
    if cf_test_df is not None and len(cf_test_df) > 0:
        print("\n[B] 因果公平评估...")

        cf_originals = cf_test_df['original_text'].tolist()
        cf_texts = cf_test_df['cf_text'].tolist()

        probs_orig = predict_batch(model, tokenizer, cf_originals, device, max_len)
        probs_cf = predict_batch(model, tokenizer, cf_texts, device, max_len)
        preds_orig = (probs_orig >= threshold).astype(int)
        preds_cf = (probs_cf >= threshold).astype(int)

        cfr = counterfactual_flip_rate(preds_orig, preds_cf)
        ctfg = counterfactual_token_fairness_gap(probs_orig, probs_cf)

        results['causal'] = {
            'cfr': float(cfr),
            'ctfg': float(ctfg),
            'n_pairs': len(cf_texts),
            'mean_prob_orig': float(np.mean(probs_orig)),
            'mean_prob_cf': float(np.mean(probs_cf)),
        }
        print(f"  CFR (翻转率):  {cfr:.4f}  (理想: 0.0)")
        print(f"  CTFG (预测差): {ctfg:.4f}  (理想: 0.0)")

        # 按群体分解
        if 'source_group' in cf_test_df.columns:
            group_cfr = {}
            for group in cf_test_df['source_group'].unique():
                g_mask = cf_test_df['source_group'].values == group
                g_cfr = counterfactual_flip_rate(preds_orig[g_mask], preds_cf[g_mask])
                g_ctfg = counterfactual_token_fairness_gap(probs_orig[g_mask], probs_cf[g_mask])
                group_cfr[group] = {'cfr': float(g_cfr), 'ctfg': float(g_ctfg),
                                     'count': int(g_mask.sum())}
            results['causal']['per_group'] = group_cfr
            print(f"  各群体 CFR:")
            for g, v in sorted(group_cfr.items(), key=lambda x: -x[1]['cfr']):
                print(f"    {g:20s}: CFR={v['cfr']:.4f}  CTFG={v['ctfg']:.4f}  (n={v['count']})")

    # ---- C. 群体公平 ----
    # 判断数据集中的群体信息
    group_col = None
    if 'target_group' in test_df.columns:
        group_col = 'target_group'
    elif 'coarse_groups' in test_df.columns:
        # HateXplain: coarse_groups 是列表，取第一个
        group_col = '_primary_group'
        test_df = test_df.copy()
        test_df['_primary_group'] = test_df['coarse_groups'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'none'
        )

    if group_col and test_df[group_col].nunique() > 1:
        print(f"\n[C] 群体公平评估 (by {group_col})...")
        groups = test_df[group_col].values
        fped, fned, group_details = compute_fped_fned(labels, preds, groups)

        results['group_fairness'] = {
            'fped': float(fped),
            'fned': float(fned),
            'per_group': group_details,
        }
        print(f"  FPED (假阳性差异): {fped:.4f}  (理想: 0.0)")
        print(f"  FNED (假阴性差异): {fned:.4f}  (理想: 0.0)")

        # Per-group F1
        per_group_f1 = {}
        for g in sorted(set(groups)):
            mask = np.array(groups) == g
            if mask.sum() > 0 and len(set(labels[mask])) > 1:
                g_f1 = f1_score(labels[mask], preds[mask], average='macro')
                per_group_f1[g] = float(g_f1)
        results['per_group_f1'] = per_group_f1
        if per_group_f1:
            print(f"  各群体 F1:")
            for g, f in sorted(per_group_f1.items(), key=lambda x: x[1]):
                print(f"    {g:20s}: F1={f:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="因果公平评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hatexplain")
    parser.add_argument("--cf_method", type=str, default="llm",
                        choices=["swap", "llm"])
    parser.add_argument("--model_name", type=str,
                        default=os.path.join(BASE_DIR, "models", "deberta-v3-base"))
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(BASE_DIR, "data", "causal_fair"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = DebertaV3CausalFair(args.model_name, num_classes=2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[Model] Loaded: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 加载测试数据
    test_df = pd.read_parquet(
        os.path.join(args.data_dir, f"{args.dataset}_test.parquet")
    )

    # 加载测试集反事实
    cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
    cf_path = os.path.join(args.data_dir, f"{args.dataset}_test_{cf_suffix}.parquet")
    cf_test_df = None
    if os.path.exists(cf_path):
        cf_test_df = pd.read_parquet(cf_path)
        print(f"[CF] 加载测试集反事实: {len(cf_test_df)} 条")

    # 评估
    results = evaluate_causal_fairness(
        model, tokenizer, test_df, cf_test_df,
        device, max_len=args.max_len, threshold=args.threshold,
    )

    # 保存
    ckpt_name = os.path.basename(args.checkpoint).replace('.pth', '')
    save_path = get_eval_path(f"{ckpt_name}_fairness.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果保存至: {save_path}")


if __name__ == "__main__":
    main()
