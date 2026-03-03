"""
TF-IDF + Logistic Regression 基线
传统文本分类强基线，用于与深度学习方法对比。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from src_script.utils.path_config import get_eval_path

def calculate_fairness_metrics(df, subgroups, prob_col):
    records = []
    for sg in subgroups:
        mask = df[sg] >= 0.5
        if mask.sum() == 0:
            continue
        sub_df = df[mask]
        y_true_sub = (sub_df['target'] >= 0.5).astype(int)
        if y_true_sub.nunique() < 2:
            continue
        subgroup_auc = roc_auc_score(y_true_sub, sub_df[prob_col])

        bpsn_mask = ((df[sg] >= 0.5) & (df['target'] < 0.5)) | ((df[sg] < 0.5) & (df['target'] >= 0.5))
        bpsn_df = df[bpsn_mask]
        y_bpsn = (bpsn_df['target'] >= 0.5).astype(int)
        bpsn_auc = roc_auc_score(y_bpsn, bpsn_df[prob_col]) if y_bpsn.nunique() >= 2 else np.nan

        bnsp_mask = ((df[sg] >= 0.5) & (df['target'] >= 0.5)) | ((df[sg] < 0.5) & (df['target'] < 0.5))
        bnsp_df = df[bnsp_mask]
        y_bnsp = (bnsp_df['target'] >= 0.5).astype(int)
        bnsp_auc = roc_auc_score(y_bnsp, bnsp_df[prob_col]) if y_bnsp.nunique() >= 2 else np.nan

        records.append({'subgroup': sg, 'subgroup_auc': subgroup_auc, 'bpsn_auc': bpsn_auc, 'bnsp_auc': bnsp_auc})
    return pd.DataFrame(records)

def main():
    print("=" * 60)
    print("TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    # 加载数据
    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    test_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "test_processed.parquet"))

    X_train = train_df['comment_text'].fillna('')
    y_train = (train_df['y_tox'] >= 0.5).astype(int)
    X_test = test_df['comment_text'].fillna('')
    y_test = (test_df['y_tox'] >= 0.5).astype(int) if 'y_tox' in test_df.columns else (test_df['target'] >= 0.5).astype(int)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF
    t0 = time.time()
    print("\n[1] Fitting TF-IDF (max_features=100000, ngram=1-2)...")
    tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), sublinear_tf=True, strip_accents='unicode')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"    TF-IDF shape: {X_train_tfidf.shape}, took {time.time()-t0:.1f}s")

    # Logistic Regression
    t0 = time.time()
    print("\n[2] Training Logistic Regression (C=1.0, max_iter=1000)...")
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1, verbose=1)
    lr.fit(X_train_tfidf, y_train)
    print(f"    Training took {time.time()-t0:.1f}s")

    # 预测
    probs = lr.predict_proba(X_test_tfidf)[:, 1]

    # 阈值扫描
    best_f1, best_thresh = 0, 0.5
    scan_history = []
    for thresh in np.arange(0.01, 1.00, 0.01):
        y_pred = (probs >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        scan_history.append({'threshold': float(thresh), 'f1': float(f1), 'precision': float(prec), 'recall': float(rec)})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred_opt = (probs >= best_thresh).astype(int)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    roc_auc = roc_auc_score(y_test, probs)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rec_curve, prec_curve)

    # 固定阈值 0.5
    y_pred_fixed = (probs >= 0.5).astype(int)
    fixed_f1 = f1_score(y_test, y_pred_fixed)
    fixed_acc = accuracy_score(y_test, y_pred_fixed)
    fixed_prec = precision_score(y_test, y_pred_fixed, zero_division=0)
    fixed_rec = recall_score(y_test, y_pred_fixed, zero_division=0)

    # Bias 指标
    identity_cols = ['male', 'female', 'black', 'white', 'muslim', 'jewish',
                     'christian', 'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness']
    test_df['model_probs'] = probs
    bias_df = calculate_fairness_metrics(test_df, identity_cols, 'model_probs')
    all_bias = bias_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.flatten()
    mean_bias_auc = float(np.nanmean(all_bias))
    worst_bias_auc = float(np.nanmin(all_bias))

    # 输出
    print(f"\n{'='*60}")
    print(f"[最优阈值 {best_thresh:.2f}] F1: {best_f1:.4f} | Acc: {acc_opt:.4f}")
    print(f"[固定阈值 0.50] F1: {fixed_f1:.4f} | Acc: {fixed_acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"Mean Bias AUC: {mean_bias_auc:.4f} | Worst-group: {worst_bias_auc:.4f}")

    print("\n各子群明细:")
    for _, row in bias_df.iterrows():
        print(f"  {row['subgroup']:35s} Sub={row['subgroup_auc']:.4f}  BPSN={row['bpsn_auc']:.4f}  BNSP={row['bnsp_auc']:.4f}")

    # 保存
    report = {
        "checkpoint": "TF-IDF + LogisticRegression (sklearn)",
        "model_type": "tfidf_lr",
        "optimal_threshold": float(best_thresh),
        "primary_metrics_optimal": {
            "threshold": float(best_thresh), "f1": float(best_f1),
            "accuracy": float(acc_opt), "roc_auc": float(roc_auc), "pr_auc": float(pr_auc)
        },
        "primary_metrics_fixed_0.5": {
            "threshold": 0.5, "f1": float(fixed_f1),
            "accuracy": float(fixed_acc), "precision": float(fixed_prec), "recall": float(fixed_rec)
        },
        "bias_metrics": {
            "mean_bias_auc": mean_bias_auc,
            "worst_group_bias_auc": worst_bias_auc,
            "per_subgroup_details": bias_df.to_dict(orient='records')
        },
        "threshold_scan_history": scan_history
    }
    os.makedirs(os.path.join(BASE_DIR, "src_result", "eval"), exist_ok=True)
    out_path = get_eval_path("TFIDF_LR_baseline_metrics.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n>>> 报告已保存: {out_path}")

if __name__ == "__main__":
    main()
