"""
公平性与任务评估指标。

提供：
- compute_cfr: 反事实翻转率
- compute_ctfg: 反事实词元公平性差距
- compute_fped_fned: 假阳性/假阴性率均等差异
- evaluate_causal_fairness: 完整评估流程
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_cfr(
    preds_orig: np.ndarray,
    preds_cf: np.ndarray,
) -> float:
    """反事实翻转率（Counterfactual Flip Rate）。

    CFR 是（原始，反事实）配对中，经身份替换后预测标签发生变化的比例。
    越低越好（0 = 完美）。

    Args:
        preds_orig: 原始文本的二值预测结果。
        preds_cf: 反事实文本的二值预测结果。

    Returns:
        CFR 值，范围 [0, 1]。
    """
    if len(preds_orig) == 0:
        return 0.0
    flips = (preds_orig != preds_cf).sum()
    return float(flips) / len(preds_orig)


def compute_ctfg(
    probs_orig: np.ndarray,
    probs_cf: np.ndarray,
) -> float:
    """反事实词元公平性差距（Counterfactual Token Fairness Gap）。

    CTFG 是原始与反事实输入之间 P(toxic) 平均绝对差值。
    越低越好（0 = 毒性分数完全一致）。

    Args:
        probs_orig: 原始文本的 P(toxic) 值。
        probs_cf: 反事实文本的 P(toxic) 值。

    Returns:
        平均绝对概率差值。
    """
    if len(probs_orig) == 0:
        return 0.0
    return float(np.mean(np.abs(probs_orig - probs_cf)))


def compute_fped_fned(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Tuple[float, float, Dict]:
    """假阳性率均等差异（FPED）与假阴性率均等差异（FNED）。

    FPED 衡量各人口群体间假阳性率的差异程度。
    FNED 对假阴性率进行相同衡量。
    两者均越低越好（0 = 各群体比率完全相等）。

    Args:
        y_true: 真实二值标签。
        y_pred: 二值预测结果。
        groups: 群体归属标签数组。

    Returns:
        (fped, fned, per_group_details)
    """
    unique_groups = sorted(set(groups))

    def fpr(y_t, y_p):
        tn_fp = (y_t == 0).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        return float(fp) / max(float(tn_fp), 1)

    def fnr(y_t, y_p):
        tp_fn = (y_t == 1).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        return float(fn) / max(float(tp_fn), 1)

    overall_fpr_val = fpr(y_true, y_pred)
    overall_fnr_val = fnr(y_true, y_pred)

    fped, fned = 0.0, 0.0
    group_details = {}

    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue
        g_fpr = fpr(y_true[mask], y_pred[mask])
        g_fnr = fnr(y_true[mask], y_pred[mask])
        fped += abs(overall_fpr_val - g_fpr)
        fned += abs(overall_fnr_val - g_fnr)
        group_details[g] = {
            "fpr": g_fpr,
            "fnr": g_fnr,
            "fpr_gap": abs(overall_fpr_val - g_fpr),
            "fnr_gap": abs(overall_fnr_val - g_fnr),
            "count": int(mask.sum()),
        }

    return fped, fned, group_details


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    texts: List[str],
    tokenizer,
    device: torch.device,
    max_len: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """批量预测毒性概率。"""
    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.cuda.amp.autocast():
            out = model(input_ids, attention_mask, return_features=False)
            probs = F.softmax(out["logits"], dim=-1)[:, 1]

        all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def evaluate_causal_fairness(
    model: torch.nn.Module,
    tokenizer,
    test_df: pd.DataFrame,
    cf_df: Optional[pd.DataFrame],
    device: torch.device,
    max_len: int = 128,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """完整的因果公平性评估流程。

    Args:
        model: 已训练的分类器。
        tokenizer: HuggingFace 分词器。
        test_df: 测试集 DataFrame（需包含 'text' 和 'binary_label' 列）。
        cf_df: 反事实测试集 DataFrame
               （需包含 'original_text'、'cf_text' 列）。
        device: 计算设备。
        max_len: 最大序列长度。
        threshold: 分类阈值。

    Returns:
        包含任务指标、因果公平性指标和群体公平性指标的字典。
    """
    results: Dict[str, Any] = {}
    texts = test_df["text"].tolist()
    labels = test_df["binary_label"].values

    # --- 任务性能指标 ---
    probs = predict_probs(model, texts, tokenizer, device, max_len)
    preds = (probs >= threshold).astype(int)

    results["task"] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "binary_f1": float(f1_score(labels, preds, average="binary")),
        "auc_roc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.5,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }

    # --- 因果公平性指标 ---
    if cf_df is not None and len(cf_df) > 0:
        cf_originals = cf_df["original_text"].tolist()
        cf_texts = cf_df["cf_text"].tolist()

        probs_orig = predict_probs(model, cf_originals, tokenizer, device, max_len)
        probs_cf = predict_probs(model, cf_texts, tokenizer, device, max_len)
        preds_orig = (probs_orig >= threshold).astype(int)
        preds_cf = (probs_cf >= threshold).astype(int)

        cfr = compute_cfr(preds_orig, preds_cf)
        ctfg = compute_ctfg(probs_orig, probs_cf)

        results["causal"] = {
            "cfr": float(cfr),
            "ctfg": float(ctfg),
            "n_pairs": len(cf_texts),
            "mean_prob_orig": float(np.mean(probs_orig)),
            "mean_prob_cf": float(np.mean(probs_cf)),
        }

        # 按群体分组统计
        if "source_group" in cf_df.columns:
            group_cfr = {}
            for group in cf_df["source_group"].unique():
                g_mask = cf_df["source_group"].values == group
                g_cfr = compute_cfr(preds_orig[g_mask], preds_cf[g_mask])
                g_ctfg = compute_ctfg(probs_orig[g_mask], probs_cf[g_mask])
                group_cfr[group] = {
                    "cfr": float(g_cfr),
                    "ctfg": float(g_ctfg),
                    "count": int(g_mask.sum()),
                }
            results["causal"]["per_group"] = group_cfr

    # --- 群体公平性指标 ---
    group_col = _detect_group_column(test_df)
    if group_col and test_df[group_col].nunique() > 1:
        groups = test_df[group_col].values
        fped, fned, group_details = compute_fped_fned(labels, preds, groups)
        results["group_fairness"] = {
            "fped": float(fped),
            "fned": float(fned),
            "per_group": group_details,
        }

    return results


def _detect_group_column(df: pd.DataFrame) -> Optional[str]:
    """检测 DataFrame 中哪一列包含人口群体标签。"""
    if "target_group" in df.columns:
        return "target_group"
    if "coarse_groups" in df.columns:
        return "coarse_groups"
    return None