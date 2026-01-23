import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

# 设置 Hugging Face 离线模式
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_deberta_mtl import DebertaToxicityMTL
from data_loader import ToxicityDataset

def calculate_auc(y_true, y_prob):
    try:
        return metrics.roc_auc_score(y_true, y_prob)
    except:
        return np.nan

def calculate_pr_auc(y_true, y_prob):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    return metrics.auc(recall, precision)

def scan_thresholds(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = metrics.f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

def calculate_fairness_metrics(df, subgroups, model_col):
    records = []
    y_true = df['target'] >= 0.5
    for subgroup in subgroups:
        sub_mask = df[subgroup] >= 0.5
        if sub_mask.sum() == 0:
            continue
            
        # Subgroup AUC
        sub_df = df[sub_mask]
        sub_auc = calculate_auc(sub_df['target'] >= 0.5, sub_df[model_col])
        
        # BPSN AUC (Background Positive, Subgroup Negative)
        bpsn_mask = ((df[subgroup] >= 0.5) & (df['target'] < 0.5)) | \
                    ((df[subgroup] < 0.5) & (df['target'] >= 0.5))
        bpsn_df = df[bpsn_mask]
        bpsn_auc = calculate_auc(bpsn_df['target'] >= 0.5, bpsn_df[model_col])
        
        # BNSP AUC (Background Negative, Subgroup Positive)
        bnsp_mask = ((df[subgroup] >= 0.5) & (df['target'] >= 0.5)) | \
                    ((df[subgroup] < 0.5) & (df['target'] < 0.5))
        bnsp_df = df[bnsp_mask]
        bnsp_auc = calculate_auc(bnsp_df['target'] >= 0.5, bnsp_df[model_col])
        
        records.append({
            'subgroup': subgroup,
            'subgroup_auc': sub_auc,
            'bpsn_auc': bpsn_auc,
            'bnsp_auc': bnsp_auc
        })
    return pd.DataFrame(records)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="deberta_mtl")
    parser.add_argument("--output_prefix", type=str, default="full_eval")
    args = parser.parse_args()

    # 模型加载逻辑 (简略版，只保留 MTL 和 Baseline)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "microsoft/deberta-v3-base" # 默认
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if args.model_type == "deberta_mtl":
        model = DebertaToxicityMTL(MODEL_PATH).to(device)
    else:
        # 兼容其他模型类型...
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1, local_files_only=True).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 数据加载
    VAL_FILE = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    val_df = pd.read_parquet(VAL_FILE)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=256)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    # 推理
    probs = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            
            logits = outputs['logits_tox'] if isinstance(outputs, dict) else outputs.logits
            probs.extend(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
            targets.extend(batch['y_tox'].cpu().numpy())

    probs = np.array(probs)
    targets = np.array(targets)
    val_df['model_probs'] = probs

    # [1] 计算主指标
    best_thresh, best_f1 = scan_thresholds(targets >= 0.5, probs)
    acc_at_best = metrics.accuracy_score(targets >= 0.5, (probs >= best_thresh).astype(int))
    pr_auc = calculate_pr_auc(targets >= 0.5, probs)
    roc_auc = calculate_auc(targets >= 0.5, probs)

    # [2] 计算偏见指标
    subgroups = ['male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian', 'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness']
    bias_df = calculate_fairness_metrics(val_df, subgroups, 'model_probs')
    
    mean_bias_auc = bias_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.mean()
    worst_bias_auc = bias_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.min()

    # 打印结果
    print("\n" + "="*40)
    print(f" EVALUATION REPORT: {args.checkpoint}")
    print("="*40)
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"F1-Score:       {best_f1:.4f}")
    print(f"Accuracy:       {acc_at_best:.4f}")
    print(f"PR-AUC:         {pr_auc:.4f}")
    print(f"ROC-AUC:        {roc_auc:.4f}")
    print("-" * 40)
    print(f"Mean Bias AUC:  {mean_bias_auc:.4f}")
    print(f"Worst Bias AUC: {worst_bias_auc:.4f}")
    print("="*40)

    # 保存结果
    results = {
        "checkpoint": args.checkpoint,
        "main_metrics": {
            "best_threshold": float(best_thresh),
            "f1": float(best_f1),
            "accuracy": float(acc_at_best),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc)
        },
        "bias_metrics": {
            "mean_bias_auc": float(mean_bias_auc),
            "worst_bias_auc": float(worst_bias_auc),
            "details": bias_df.to_dict(orient='records')
        }
    }
    
    output_path = os.path.join(BASE_DIR, "src_result", f"{args.output_prefix}_metrics.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
