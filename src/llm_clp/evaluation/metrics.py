"""
Fairness and task evaluation metrics.

Provides:
- compute_cfr: Counterfactual Flip Rate
- compute_ctfg: Counterfactual Token Fairness Gap
- compute_fped_fned: False Positive/Negative Rate Equalized Difference
- evaluate_causal_fairness: full evaluation pipeline
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
    """Counterfactual Flip Rate.

    CFR is the fraction of (original, counterfactual) pairs where the predicted
    label changes after identity substitution. Lower is better (0 = perfect).

    Args:
        preds_orig: Binary predictions for original texts.
        preds_cf: Binary predictions for counterfactual texts.

    Returns:
        CFR in [0, 1].
    """
    if len(preds_orig) == 0:
        return 0.0
    flips = (preds_orig != preds_cf).sum()
    return float(flips) / len(preds_orig)


def compute_ctfg(
    probs_orig: np.ndarray,
    probs_cf: np.ndarray,
) -> float:
    """Counterfactual Token Fairness Gap.

    CTFG is the mean absolute difference in P(toxic) between original and
    counterfactual inputs. Lower is better (0 = identical toxicity scores).

    Args:
        probs_orig: P(toxic) for original texts.
        probs_cf: P(toxic) for counterfactual texts.

    Returns:
        Mean absolute probability difference.
    """
    if len(probs_orig) == 0:
        return 0.0
    return float(np.mean(np.abs(probs_orig - probs_cf)))


def compute_fped_fned(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Tuple[float, float, Dict]:
    """False Positive Rate Equalized Difference (FPED) and
    False Negative Rate Equalized Difference (FNED).

    FPED measures how much false positive rates differ across demographic groups.
    FNED measures the same for false negative rates.
    Both lower is better (0 = equal rates across all groups).

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Binary predictions.
        groups: Array of group membership labels.

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
    """Batch predict toxic probabilities."""
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
    """Full causal fairness evaluation pipeline.

    Args:
        model: Trained classifier.
        tokenizer: HuggingFace tokenizer.
        test_df: Test set DataFrame (needs 'text' and 'binary_label' columns).
        cf_df: Counterfactual test set DataFrame
               (needs 'original_text', 'cf_text' columns).
        device: Compute device.
        max_len: Max sequence length.
        threshold: Classification threshold.

    Returns:
        Dict with task metrics, causal fairness metrics, and group fairness metrics.
    """
    results: Dict[str, Any] = {}
    texts = test_df["text"].tolist()
    labels = test_df["binary_label"].values

    # --- Task Metrics ---
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

    # --- Causal Fairness Metrics ---
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

        # Per-group breakdown
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

    # --- Group Fairness ---
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
    """Detect which column contains demographic group labels."""
    if "target_group" in df.columns:
        return "target_group"
    if "coarse_groups" in df.columns:
        return "_primary_group"
    return None