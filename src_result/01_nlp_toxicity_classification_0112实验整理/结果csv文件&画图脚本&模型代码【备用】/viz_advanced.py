import os
import sys

# Setup HF Environment FIRST
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer

# Configure Fonts and Styles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 12

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
from model_deberta_mtl import DebertaToxicityMTL

RESULT_DIR = os.path.join(BASE_DIR, "src_result")
FIGURE_DIR = os.path.join(BASE_DIR, "src_result", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_ablation_bar_chart():
    """Figure 5: Grouped Bar Chart for Ablation Study"""
    print("Generating Figure 5: Ablation Bar Chart...")
    
    variants = [
        {"name": "Full Model", "file_f1": "metrics_f1_Proposed_DeBERTa_V3_MTL_Reweight.csv", "file_fair": "metrics_fair_Proposed_DeBERTa_V3_MTL_Reweight.csv"},
        {"name": "w/o Reweight", "file_f1": "metrics_f1_Ablation_No_Reweight.csv", "file_fair": "metrics_fair_Ablation_No_Reweight.csv"},
        {"name": "w/o MTL", "file_f1": "metrics_f1_Ablation_Single_Task.csv", "file_fair": "metrics_fair_Ablation_Single_Task.csv"},
        {"name": "w/o Pooling", "file_f1": "metrics_f1_Ablation_CLS_Only.csv", "file_fair": "metrics_fair_Ablation_CLS_Only.csv"}
    ]
    
    data = []
    for v in variants:
        try:
            df_f1 = pd.read_csv(os.path.join(RESULT_DIR, v["file_f1"]))
            df_fair = pd.read_csv(os.path.join(RESULT_DIR, v["file_fair"]))
            data.append({
                "Variant": v["name"],
                "F1 Score": df_f1['f1'].max(),
                "Mean BPSN AUC": df_fair['bpsn_auc'].mean()
            })
        except: pass
    
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars="Variant", var_name="Metric", value_name="Value")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x="Variant", y="Value", hue="Metric", palette="muted")
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
    
    plt.ylim(0.6, 1.0)
    plt.title("Ablation Study: Impact of Components", fontsize=15, weight='bold')
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(FIGURE_DIR, "Fig5_Ablation_Bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sensitivity_analysis():
    """Figure 6: Threshold Sensitivity Analysis"""
    print("Generating Figure 6: Threshold Sensitivity...")
    
    df = pd.read_csv(os.path.join(RESULT_DIR, "metrics_f1_Proposed_DeBERTa_V3_MTL_Reweight.csv"))
    
    plt.figure(figsize=(9, 6))
    plt.plot(df['threshold'], df['f1'], marker='o', label='F1 Score', color='#2ecc71', linewidth=2)
    plt.plot(df['threshold'], df['accuracy'], marker='s', label='Accuracy', color='#3498db', linewidth=1.5, alpha=0.7)
    
    # Highlight optimal point
    best_row = df.loc[df['f1'].idxmax()]
    plt.axvline(best_row['threshold'], color='red', linestyle='--', alpha=0.5)
    plt.text(best_row['threshold']+0.02, 0.75, f"Optimal T={best_row['threshold']}", color='red', weight='bold')
    
    plt.xlabel("Threshold")
    plt.ylabel("Performance")
    plt.title("Threshold Sensitivity Analysis (Proposed Model)", fontsize=15, weight='bold')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(FIGURE_DIR, "Fig6_Threshold_Sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_subgroup_heatmap():
    """Figure 7: Subgroup Performance Heatmap"""
    print("Generating Figure 7: Subgroup Heatmap...")
    
    df = pd.read_csv(os.path.join(RESULT_DIR, "metrics_fair_Proposed_DeBERTa_V3_MTL_Reweight.csv"))
    df = df.set_index('subgroup')
    
    # Focus on AUC metrics
    metrics_to_plot = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    plot_df = df[metrics_to_plot]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
    plt.title("Detailed Performance Matrix across Demographic Groups", fontsize=14, weight='bold')
    plt.savefig(os.path.join(FIGURE_DIR, "Fig7_Subgroup_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_heatmap():
    """Figure 8: Attention Interpretability Heatmap"""
    print("Generating Figure 8: Attention Heatmap...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", local_files_only=True)
    model = DebertaToxicityMTL("microsoft/deberta-v3-base", use_attention_pooling=True)
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "res_stage2_final.pth"), map_location=device))
    model.to(device).eval()
    
    # Example sentence that often triggers bias but is non-toxic
    text = "As a proud black man, I believe in community and respect."
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model.deberta(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        last_hidden_state = outputs.last_hidden_state
        
        # Manually calculate attention scores from our pooling layer
        # Score shape: (batch_size, seq_len, 1)
        score = model.pooling.score_layer(last_hidden_state)
        # Weight shape: (batch_size, seq_len)
        weights = torch.softmax(score.squeeze(-1), dim=1).cpu().numpy()[0]
        
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # Remove special tokens for cleaner visualization if needed, but keeping them shows model behavior
    
    plt.figure(figsize=(12, 3))
    df_attn = pd.DataFrame([weights], columns=tokens)
    sns.heatmap(df_attn, annot=True, fmt=".2f", cmap="Reds", cbar=False)
    plt.title("Attention Weights on Non-Toxic Identity-related Sentence", fontsize=14, weight='bold')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(FIGURE_DIR, "Fig8_Attention_Weights.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_ablation_bar_chart()
    plot_sensitivity_analysis()
    plot_subgroup_heatmap()
    try:
        plot_attention_heatmap()
    except Exception as e:
        print(f"Error in Fig 8: {e}")
    
    print(f"Advanced figures saved to {FIGURE_DIR}")
