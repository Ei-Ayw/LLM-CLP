import os
import sys

# Setup HF Environment FIRST
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import torch
from transformers import AutoTokenizer
from sklearn.manifold import TSNE

# Configure Fonts and Styles for SCI Papers
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # Standard SCI fonts
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

from model_deberta_mtl import DebertaToxicityMTL
from data_loader import ToxicityDataset

RESULT_DIR = os.path.join(BASE_DIR, "src_result")
FIGURE_DIR = os.path.join(BASE_DIR, "src_result", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_radar_chart():
    """Figure 1: Fairness Radar Chart (Comparison of BPSN AUC)"""
    print("Generating Figure 1: Radar Chart...")
    
    # Load data
    df_bert = pd.read_csv(os.path.join(RESULT_DIR, "metrics_fair_Baseline_BERT_Base.csv"))
    df_plain = pd.read_csv(os.path.join(RESULT_DIR, "metrics_fair_Ablation_Single_Task.csv"))
    df_prop = pd.read_csv(os.path.join(RESULT_DIR, "metrics_fair_Proposed_DeBERTa_V3_MTL_Reweight.csv"))
    
    label_map = {
        'male': 'Male', 'female': 'Female', 'black': 'Black', 
        'white': 'White', 'muslim': 'Muslim', 
        'homosexual_gay_or_lesbian': 'Homosexual',
        'psychiatric_or_mental_illness': 'Psychiatric'
    }
    
    categories = list(label_map.values())
    N = len(categories)
    
    def get_values(df):
        vals = []
        for key in label_map.keys():
            vals.append(df[df['subgroup'] == key]['bpsn_auc'].values[0])
        vals += [vals[0]]
        return vals

    val_bert = get_values(df_bert)
    val_plain = get_values(df_plain)
    val_prop = get_values(df_prop)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Comparison (BERT)
    ax.plot(angles, val_bert, linewidth=1.5, linestyle=':', color='grey', label='Comparison (BERT)')
    
    # Baseline (DeBERTa-V3 Vanilla)
    ax.plot(angles, val_plain, linewidth=2, linestyle='dashed', color='#3498db', label='Baseline (DeBERTa-V3 Vanilla)')
    ax.fill(angles, val_plain, '#3498db', alpha=0.05)
    
    # Proposed
    ax.plot(angles, val_prop, linewidth=3, linestyle='solid', color='#2ecc71', label='Proposed (MTL + Reweight)')
    ax.fill(angles, val_prop, '#2ecc71', alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, weight='bold')
    ax.set_ylim(0.80, 1.0)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
    plt.title("Bias Mitigation: Comparison vs Baseline vs Proposed", size=16, weight='bold', y=1.1)
    
    plt.savefig(os.path.join(FIGURE_DIR, "Fig1_Radar_Fairness.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_scatter():
    """Figure 2: Performance vs Fairness Scatter Plot"""
    print("Generating Figure 2: Pareto Scatter Plot...")
    
    models = [
        {"name": "Comparison (BERT)", "file_f1": "metrics_f1_Baseline_BERT_Base.csv", "file_fair": "metrics_fair_Baseline_BERT_Base.csv"},
        {"name": "Comparison (RoBERTa)", "file_f1": "metrics_f1_Baseline_RoBERTa_Base.csv", "file_fair": "metrics_fair_Baseline_RoBERTa_Base.csv"},
        {"name": "Baseline (DeBERTa-V3 Vanilla)", "file_f1": "metrics_f1_Ablation_Single_Task.csv", "file_fair": "metrics_fair_Ablation_Single_Task.csv"},
        {"name": "Proposed (Full)", "file_f1": "metrics_f1_Proposed_DeBERTa_V3_MTL_Reweight.csv", "file_fair": "metrics_fair_Proposed_DeBERTa_V3_MTL_Reweight.csv"}
    ]
    
    data = []
    for m in models:
        try:
            df_f1 = pd.read_csv(os.path.join(RESULT_DIR, m["file_f1"]))
            df_fair = pd.read_csv(os.path.join(RESULT_DIR, m["file_fair"]))
            data.append({"Model": m["name"], "F1 Score": df_f1['f1'].max(), "Mean BPSN AUC": df_fair['bpsn_auc'].mean()})
        except: pass
            
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="Mean BPSN AUC", y="F1 Score", s=300, hue="Model", style="Model", palette="viridis")
    
    for i, row in df.iterrows():
        plt.text(row["Mean BPSN AUC"]+0.001, row["F1 Score"]+0.001, row["Model"], fontsize=11)
        
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title("Fairness-Accuracy Pareto Frontier", fontsize=16, weight='bold')
    plt.savefig(os.path.join(FIGURE_DIR, "Fig2_Pareto_Scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_density():
    """Figure 3: Probability Density Comparison on Challenging Samples"""
    print("Generating Figure 3: Probability Density Plot...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_path = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    df = pd.read_parquet(val_path)
    
    identity_cols = ['black', 'white', 'muslim', 'homosexual_gay_or_lesbian']
    mask = (df[identity_cols].max(axis=1) >= 0.5) & (df['y_tox'] == 0)
    subset_df = df[mask].sample(n=500, random_state=42)
    
    # Load Baseline (DeBERTa Vanilla / Single Task)
    model_base = DebertaToxicityMTL("microsoft/deberta-v3-base", use_attention_pooling=True)
    model_base.load_state_dict(torch.load(os.path.join(RESULT_DIR, "res_ablation_no_mtl.pth"), map_location=device))
    model_base.to(device).eval()
    
    # Load Proposed
    model_prop = DebertaToxicityMTL("microsoft/deberta-v3-base", use_attention_pooling=True)
    model_prop.load_state_dict(torch.load(os.path.join(RESULT_DIR, "res_stage2_final.pth"), map_location=device))
    model_prop.to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", local_files_only=True)
    
    probs_base = []
    probs_prop = []
    
    texts = subset_df['comment_text'].tolist()
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            
            out_b = model_base(inputs['input_ids'], inputs['attention_mask'])
            probs_base.extend(torch.sigmoid(out_b['logits_tox']).cpu().flatten().numpy())
            
            out_p = model_prop(inputs['input_ids'], inputs['attention_mask'])
            probs_prop.extend(torch.sigmoid(out_p['logits_tox']).cpu().flatten().numpy())
            
    plt.figure(figsize=(10, 6))
    sns.kdeplot(probs_base, label='Baseline (DeBERTa-V3 Vanilla)', fill=True, color='#3498db')
    sns.kdeplot(probs_prop, label='Proposed (Full Model)', fill=True, color='#2ecc71')
    
    plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold=0.5')
    plt.title("Suppressing False Positives in Identity-related Content", size=15, weight='bold')
    plt.xlabel("Predicted Toxicity Probability (Target=0 Samples)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(FIGURE_DIR, "Fig3_Probability_Density.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne():
    """Figure 4: t-SNE of Embeddings"""
    print("Generating Figure 4: t-SNE Embedding Plot...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data (Mixed Toxic and Non-toxic, ALL containing specific identity e.g., 'gay' or 'black')
    val_path = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    df = pd.read_parquet(val_path)
    
    # Focus on one sensitive identity for clear visualization
    target_identity = 'black'
    
    mask_identity = df[target_identity] >= 0.5
    # Balance toxic and non-toxic
    df_pos = df[mask_identity & (df['y_tox'] == 1)].sample(200, random_state=42)
    df_neg = df[mask_identity & (df['y_tox'] == 0)].sample(200, random_state=42)
    df_viz = pd.concat([df_pos, df_neg])
    
    # Labels
    labels = ["Toxic" if x==1 else "Non-Toxic" for x in df_viz['y_tox']]
    
    # Get Embeddings from Proposed Model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", local_files_only=True)
    model = DebertaToxicityMTL("microsoft/deberta-v3-base", use_attention_pooling=True)
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "res_stage2_final.pth"), map_location=device))
    model.to(device)
    model.eval()
    
    embeddings = []
    texts = df_viz['comment_text'].tolist()
    
    # We need to hook into the model to get the feature vector 'z' (before classification)
    # Or just re-implement the forward pass part
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch_texts = texts[i:i+16]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            
            # Forward partially
            outputs = model.deberta(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            last_hidden_state = outputs.last_hidden_state
            
            # Repro the pooling logic
            h_cls = last_hidden_state[:, 0, :]
            h_att = model.pooling(last_hidden_state, inputs['attention_mask'])
            h = torch.cat([h_cls, h_att], dim=-1)
            
            # Projection
            z = model.projection(h) # (B, 512)
            embeddings.append(z.cpu().numpy())
            
    X = np.concatenate(embeddings, axis=0)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels, palette={"Toxic": "#e74c3c", "Non-Toxic": "#2ecc71"}, s=100, alpha=0.8)
    
    plt.title(f"Feature Space Visualization (Identity: '{target_identity}')", size=18, weight='bold')
    plt.axis('off') # Cleaner look
    
    plt.savefig(os.path.join(FIGURE_DIR, "Fig4_TSNE_Embeddings.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_radar_chart()
    plot_pareto_scatter()
    try:
        plot_probability_density()
    except Exception as e:
        print(f"Error generating Fig 3: {e}")
        
    try:
        plot_tsne()
    except Exception as e:
        print(f"Error generating Fig 4: {e}")
        
    print(f"All figures saved to {FIGURE_DIR}")
