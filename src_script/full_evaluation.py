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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR); sys.path.append(os.path.join(BASE_DIR, "src_model")); sys.path.append(os.path.join(BASE_DIR, "src_script"))

# 离线环境
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 导入所有物理模型定义
from model_deberta_v3_mtl import DebertaV3MTL
from model_bert_cnn_bilstm import BertCNNBiLSTM
from model_text_cnn import TextCNN
from model_bilstm import BiLSTM
from model_vanilla_bert import VanillaBERT
from model_vanilla_roberta import VanillaRoBERTa
from model_vanilla_deberta_v3 import VanillaDeBERTaV3
from data_loader import ToxicityDataset

class SimpleTokenDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return {'ids': torch.tensor(ids, dtype=torch.long), 'y_tox': torch.tensor(self.labels[idx], dtype=torch.float)}

def calculate_auc(y_true, y_prob):
    try: return metrics.roc_auc_score(y_true, y_prob)
    except: return np.nan

def scan_thresholds(y_true, y_prob):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.05, 0.96, 0.05):
        f1 = metrics.f1_score(y_true, (y_prob >= thresh).astype(int))
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
        bpsn_auc = calculate_auc(df[bpsn_mask]['target'] >= 0.5, df[bpsn_mask][model_col])
        bnsp_mask = ((df[subgroup] >= 0.5) & (df['target'] >= 0.5)) | ((df[subgroup] < 0.5) & (df['target'] < 0.5))
        bnsp_auc = calculate_auc(df[bnsp_mask]['target'] >= 0.5, df[bnsp_mask][model_col])
        records.append({'subgroup': subgroup, 'subgroup_auc': sub_auc, 'bpsn_auc': bpsn_auc, 'bnsp_auc': bnsp_auc})
    return pd.DataFrame(records)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["deberta_mtl", "bert_cnn", "text_cnn", "bilstm", "vanilla_bert", "vanilla_roberta", "vanilla_deberta"])
    parser.add_argument("--output_prefix", type=str, default="full_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    
    ckpt_name = os.path.basename(args.checkpoint)

    # [1] 加载模型逻辑：自动识别消融开关 (NoPooling)
    if args.model_type == "deberta_mtl":
        # 核心逻辑：如果文件名包含 NoPooling，则实例化无池化版的模型架构
        use_pool = "NoPooling" not in ckpt_name
        model = DebertaV3MTL(use_attention_pooling=use_pool).to(device)
    elif args.model_type == "bert_cnn": model = BertCNNBiLSTM().to(device)
    elif args.model_type == "vanilla_bert": model = VanillaBERT().to(device)
    elif args.model_type == "vanilla_roberta": model = VanillaRoBERTa().to(device)
    elif args.model_type == "vanilla_deberta": model = VanillaDeBERTaV3().to(device)
    elif args.model_type in ["text_cnn", "bilstm"]:
        with open(args.checkpoint.replace(".pth", "_vocab.pkl"), 'rb') as f: vocab = pickle.load(f)
        model = TextCNN(vocab_size=len(vocab.stoi)).to(device) if args.model_type == "text_cnn" else BiLSTM(vocab_size=len(vocab.stoi)).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # [2] 推理流程
    probs, targets = [], []
    if args.model_type in ["text_cnn", "bilstm"]:
        loader = DataLoader(SimpleTokenDataset(val_df['comment_text'].values, val_df['y_tox'].values, vocab), batch_size=64)
        with torch.no_grad():
            for b in tqdm(loader, desc="Eval Classic"):
                out = model(b['ids'].to(device))
                probs.extend(torch.sigmoid(out['logits_tox']).squeeze(-1).cpu().numpy())
                targets.extend(b['y_tox'].cpu().numpy())
    else:
        base_name = "microsoft/deberta-v3-base" if "Deberta" in ckpt_name else "roberta-base" if "RoBERTa" in ckpt_name else "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(base_name, local_files_only=True)
        loader = DataLoader(ToxicityDataset(val_df, tokenizer), batch_size=16)
        with torch.no_grad():
            for b in tqdm(loader, desc="Eval Transformer"):
                out = model(b['input_ids'].to(device), b['attention_mask'].to(device))
                probs.extend(torch.sigmoid(out['logits_tox']).squeeze(-1).cpu().numpy())
                targets.extend(b['y_tox'].cpu().numpy())

    probs, targets = np.array(probs), np.array(targets)
    val_df['m_probs'] = probs
    best_thresh, best_f1 = scan_thresholds(targets >= 0.5, probs)
    bias_df = calculate_fairness_metrics(val_df, ['male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian', 'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'], 'm_probs')

    report = {
        "checkpoint": args.checkpoint, "f1": float(best_f1), "roc_auc": float(calculate_auc(targets >= 0.5, probs)),
        "mean_bias_auc": float(bias_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.mean())
    }
    with open(os.path.join(BASE_DIR, "src_result", f"{args.output_prefix}_metrics.json"), 'w') as f: json.dump(report, f, indent=4)
    print(f"\n[FINISH] Result saved for: {ckpt_name}")

if __name__ == "__main__":
    main()
