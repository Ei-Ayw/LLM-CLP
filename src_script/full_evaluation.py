import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import pickle

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

from model_deberta_mtl import DebertaV3MTL
from model_bert_cnn_bilstm import BertCNNBiLSTM
from model_classic_deep import TextCNN, BiLSTM
from model_vanilla_transformers import VanillaTransformer
from data_loader import ToxicityDataset

# --- 辅助：针对经典模型的 Dataset 包装 ---
class SimpleTokenDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return {'ids': torch.tensor(ids, dtype=torch.long), 'y_tox': torch.tensor(self.labels[idx], dtype=torch.float)}

def calculate_auc(y_true, y_prob):
    try: return metrics.roc_auc_score(y_true, y_prob)
    except: return np.nan

def calculate_pr_auc(y_true, y_prob):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    return metrics.auc(recall, precision)

def scan_thresholds(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = metrics.f1_score(y_true, y_pred)
        if f1 > best_f1: best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1

def calculate_fairness_metrics(df, subgroups, model_col):
    records = []
    for subgroup in subgroups:
        sub_mask = df[subgroup] >= 0.5
        if sub_mask.sum() == 0: continue
        sub_df = df[sub_mask]
        sub_auc = calculate_auc(sub_df['target'] >= 0.5, sub_df[model_col])
        bpsn_mask = ((df[subgroup] >= 0.5) & (df['target'] < 0.5)) | ((df[subgroup] < 0.5) & (df['target'] >= 0.5))
        bpsn_df = df[bpsn_mask]
        bpsn_auc = calculate_auc(bpsn_df['target'] >= 0.5, bpsn_df[model_col])
        bnsp_mask = ((df[subgroup] >= 0.5) & (df['target'] >= 0.5)) | ((df[subgroup] < 0.5) & (df['target'] < 0.5))
        bnsp_df = df[bnsp_mask]
        bnsp_auc = calculate_auc(bnsp_df['target'] >= 0.5, bnsp_df[model_col])
        records.append({'subgroup': subgroup, 'subgroup_auc': sub_auc, 'bpsn_auc': bpsn_auc, 'bnsp_auc': bnsp_auc})
    return pd.DataFrame(records)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["deberta_mtl", "bert_cnn", "text_cnn", "bilstm", "vanilla"])
    parser.add_argument("--output_prefix", type=str, default="full_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAL_FILE = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    val_df = pd.read_parquet(VAL_FILE)
    
    # [1] 根据模型类型加载模型与权重
    if args.model_type == "deberta_mtl":
        model = DebertaV3MTL("microsoft/deberta-v3-base").to(device)
    elif args.model_type == "bert_cnn":
        model = BertCNNBiLSTM("bert-base-uncased").to(device)
    elif args.model_type == "vanilla":
        model = VanillaTransformer("bert-base-uncased").to(device) # 默认以 BERT 评估，如有需要可扩展参数
    elif args.model_type in ["text_cnn", "bilstm"]:
        vocab_path = args.checkpoint.replace(".pth", "_vocab.pkl")
        with open(vocab_path, 'rb') as f: vocab = pickle.load(f)
        vocab_size = len(vocab.stoi)
        if args.model_type == "text_cnn": model = TextCNN(vocab_size=vocab_size).to(device)
        else: model = BiLSTM(vocab_size=vocab_size).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # [2] 推理逻辑
    probs, targets = [], []
    if args.model_type in ["text_cnn", "bilstm"]:
        ds = SimpleTokenDataset(val_df['comment_text'].values, val_df['y_tox'].values, vocab)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating (Classic)"):
                ids = batch['ids'].to(device)
                out = model(ids)
                probs.extend(torch.sigmoid(out['logits_tox']).squeeze(-1).cpu().numpy())
                targets.extend(batch['y_tox'].cpu().numpy())
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base" if "deberta" in args.checkpoint else "bert-base-uncased", local_files_only=True)
        ds = ToxicityDataset(val_df, tokenizer)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating (Transformer)"):
                out = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                probs.extend(torch.sigmoid(out['logits_tox']).squeeze(-1).cpu().numpy())
                targets.extend(batch['y_tox'].cpu().numpy())

    probs, targets = np.array(probs), np.array(targets)
    val_df['model_probs'] = probs

    # [3] 计算指标
    best_thresh, best_f1 = scan_thresholds(targets >= 0.5, probs)
    res = {
        "checkpoint": args.checkpoint,
        "main_metrics": {
            "f1": float(best_f1), "accuracy": float(metrics.accuracy_score(targets >= 0.5, (probs >= best_thresh).astype(int))),
            "pr_auc": float(calculate_pr_auc(targets >= 0.5, probs)), "roc_auc": float(calculate_auc(targets >= 0.5, probs))
        }
    }
    bias_df = calculate_fairness_metrics(val_df, ['male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian', 'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'], 'model_probs')
    res["bias_metrics"] = {"mean_bias_auc": float(bias_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.mean()),
                           "worst_bias_auc": float(bias_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.min()),
                           "details": bias_df.to_dict(orient='records')}

    output_path = os.path.join(BASE_DIR, "src_result", f"{args.output_prefix}_metrics.json")
    with open(output_path, 'w', encoding='utf-8') as f: json.dump(res, f, indent=4, ensure_ascii=False)
    print(f"\nEvaluation Report:\nF1: {res['main_metrics']['f1']:.4f}, Mean Bias AUC: {res['bias_metrics']['mean_bias_auc']:.4f}")

if __name__ == "__main__":
    main()
