import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =============================================================================
# ### 可视化脚本：viz_feature_t_sne.py (3x3 多子图版) ###
# 设计说明：
# 本脚本为论文生成 3x3 多子图 t-SNE 可视化，展示模型在 9 个身份群体上的
# 特征空间分布。这种展示方式更有学术说服力，能清晰展示模型的公平性。
# =============================================================================

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script", "data"))

# 设置 Hugging Face 离线模型
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_deberta_v3_mtl import DebertaV3MTL
from exp_data_loader import ToxicityDataset

# 9 个身份群体 (对应论文的评估标准)
IDENTITY_COLS = [
    'male', 'female', 'black', 'white', 'muslim', 
    'jewish', 'christian', 'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
]

# 简化标签名 (用于子图标题)
IDENTITY_LABELS = [
    'Male', 'Female', 'Black', 'White', 'Muslim',
    'Jewish', 'Christian', 'LGBTQ+', 'Mental Illness'
]

def extract_features_for_subgroup(model, df, tokenizer, subgroup, n_samples=200, device='cuda'):
    """为单个身份子群提取特征"""
    sub_df = df[df[subgroup] >= 0.5].copy()
    if len(sub_df) == 0:
        return None, None
    
    # 平衡采样
    toxic_df = sub_df[sub_df['target'] >= 0.5]
    nontoxic_df = sub_df[sub_df['target'] < 0.5]
    
    n_per_class = n_samples // 2
    toxic_sample = toxic_df.sample(min(len(toxic_df), n_per_class), random_state=42)
    nontoxic_sample = nontoxic_df.sample(min(len(nontoxic_df), n_per_class), random_state=42)
    
    plot_df = pd.concat([toxic_sample, nontoxic_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 提取特征
    ds = ToxicityDataset(plot_df, tokenizer, max_len=128)  # 短一点加速
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            features.append(outputs['features'].cpu().numpy())
            labels.extend((batch['y_tox'] >= 0.5).numpy().astype(int))
    
    return np.concatenate(features, axis=0), np.array(labels)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--n_samples", type=int, default=200, help="每个子群采样数")
    parser.add_argument("--output_name", type=str, default="tsne_9_identities.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "microsoft/deberta-v3-base"
    
    print(">>> 加载模型和数据...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = DebertaV3MTL(MODEL_PATH).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 使用测试集进行可视化
    df = pd.read_parquet(os.path.join(BASE_DIR, "data", "test_processed.parquet"))
    
    # =========================================================================
    # 创建 3x3 多子图
    # =========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    colors = {"Non-Toxic": "#3CB371", "Toxic": "#FF7F50"}
    
    for idx, (identity, label) in enumerate(zip(IDENTITY_COLS, IDENTITY_LABELS)):
        print(f">>> 处理身份群体: {label} ({idx+1}/9)")
        ax = axes[idx]
        
        features, labels = extract_features_for_subgroup(
            model, df, tokenizer, identity, args.n_samples, device
        )
        
        if features is None or len(features) < 10:
            ax.text(0.5, 0.5, f'{label}\n(数据不足)', ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=min(30, len(features)//3), n_iter=500, random_state=42)
        feat_2d = tsne.fit_transform(features)
        
        # 绘制散点图
        for i, (class_label, color) in enumerate(colors.items()):
            mask = (labels == i)
            ax.scatter(feat_2d[mask, 0], feat_2d[mask, 1], 
                      c=color, label=class_label, alpha=0.7, edgecolors='w', s=40)
        
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    plt.suptitle('Feature Space Visualization Across Identity Groups', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, "src_result", args.output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> 3x3 多子图已保存: {output_path}")

if __name__ == "__main__":
    main()
