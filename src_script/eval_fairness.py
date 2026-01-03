import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

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

def calculate_overall_auc(df, model_col):
    true_labels = df['target'] >= 0.5
    predicted_labels = df[model_col]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def calculate_subgroup_auc(df, subgroup, model_col):
    subgroup_examples = df[df[subgroup] >= 0.5]
    true_labels = subgroup_examples['target'] >= 0.5
    predicted_labels = subgroup_examples[model_col]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def calculate_bpsn_auc(df, subgroup, model_col):
    """
    Background Positive, Subgroup Negative.
    Subset: Subgroup Negative + Background Positive
    """
    subgroup_negative_examples = df[(df[subgroup] >= 0.5) & (df['target'] < 0.5)]
    background_positive_examples = df[(df[subgroup] < 0.5) & (df['target'] >= 0.5)]
    bpsn_subset = pd.concat([subgroup_negative_examples, background_positive_examples])
    true_labels = bpsn_subset['target'] >= 0.5
    predicted_labels = bpsn_subset[model_col]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def calculate_bnsp_auc(df, subgroup, model_col):
    """
    Background Negative, Subgroup Positive.
    Subset: Subgroup Positive + Background Negative
    """
    subgroup_positive_examples = df[(df[subgroup] >= 0.5) & (df['target'] >= 0.5)]
    background_negative_examples = df[(df[subgroup] < 0.5) & (df['target'] < 0.5)]
    bnsp_subset = pd.concat([subgroup_positive_examples, background_negative_examples])
    true_labels = bnsp_subset['target'] >= 0.5
    predicted_labels = bnsp_subset[model_col]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def calculate_bias_metrics_for_model(dataset, subgroups, model_col):
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_auc': calculate_subgroup_auc(dataset, subgroup, model_col),
            'bpsn_auc': calculate_bpsn_auc(dataset, subgroup, model_col),
            'bnsp_auc': calculate_bnsp_auc(dataset, subgroup, model_col)
        }
        records.append(record)
    return pd.DataFrame(records)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="deberta_mtl", choices=["deberta_mtl", "deberta_cls", "bert_cnn"])
    parser.add_argument("--output_name", type=str, default="res_fairness_metrics.csv")
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
    
    # Inference
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference for Fairness"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            
            if isinstance(outputs, dict):
                logits = outputs['logits_tox']
            else: # AutoModel output
                logits = outputs.logits
                
            probs.extend(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
            
    val_df['model_probs'] = probs
    
    subgroups = [
        'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian', 
        'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
    ]
    
    # We use 'y_tox' as our ground truth target here strictly >= 0.5
    # Wait, the calculation functions use 'target' >= 0.5. 
    # In my processed df, 'target' is the original raw score (0-1).
    # ensure 'target' exists. Parquet should have it.
    
    bias_metrics_df = calculate_bias_metrics_for_model(val_df, subgroups, 'model_probs')
    
    print("\nBias Metrics per Subgroup:")
    print(bias_metrics_df)
    
    # Summary scores
    mean_bias_auc = bias_metrics_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.mean()
    worst_bias_auc = bias_metrics_df[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].values.min()
    
    print(f"\nMean Bias AUC: {mean_bias_auc:.4f}")
    print(f"Worst Group Bias AUC: {worst_bias_auc:.4f}")
    
    # Save results
    output_path = os.path.join(BASE_DIR, "src_result", args.output_name)
    bias_metrics_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
