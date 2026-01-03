import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc

# Set project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

# Set Hugging Face cache directory and mirror endpoint
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model_deberta_mtl import DebertaToxicityMTL
from data_loader import ToxicityDataset

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y_tox = batch['y_tox'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            if isinstance(outputs, dict):
                logits = outputs['logits_tox'].squeeze(-1)
            else: # AutoModel output
                logits = outputs.logits.squeeze(-1)
                
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y_tox.cpu().numpy())
            
    return np.array(all_preds), np.array(all_targets)

def scan_thresholds(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    
    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        results.append({'threshold': thresh, 'f1': f1, 'accuracy': acc})
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1, pd.DataFrame(results)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="deberta_mtl", choices=["deberta_mtl", "deberta_cls", "bert_cnn"])
    parser.add_argument("--output_name", type=str, default="res_threshold_metrics.csv")
    args = parser.parse_args()

    MODEL_PATH = "microsoft/deberta-v3-base" if "deberta" in args.model_type else "bert-base-uncased"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VAL_FILE = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    BATCH_SIZE = 16
    MAX_LEN = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if args.model_type == "deberta_mtl":
        model = DebertaToxicityMTL(MODEL_PATH, use_attention_pooling=True).to(device)
    elif args.model_type == "deberta_cls":
        model = DebertaToxicityMTL(MODEL_PATH, use_attention_pooling=False).to(device)
    elif args.model_type == "bert_cnn":
        from model_baselines import BertCNNBiLSTM
        model = BertCNNBiLSTM(MODEL_PATH).to(device)
    else: # bert_base, roberta_base, deberta_base
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("Error: Checkpoint not found.")
        return

    val_df = pd.read_parquet(VAL_FILE)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=MAX_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    probs, targets = get_predictions(model, val_loader, device)
    
    best_thresh, best_f1, results_df = scan_thresholds(targets, probs)
    print(f"\nBest Threshold: {best_thresh:.2f}, Best F1: {best_f1:.4f}")
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(targets, probs)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Final Accuracy at best threshold
    final_acc = accuracy_score(targets, (probs >= best_thresh).astype(int))
    print(f"Accuracy at Best Threshold: {final_acc:.4f}")
    
    # Save results
    output_path = os.path.join(BASE_DIR, "src_result", args.output_name)
    results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
