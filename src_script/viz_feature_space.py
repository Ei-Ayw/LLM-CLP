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

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

# 设置 Hugging Face 离线模型
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_deberta_mtl import DebertaToxicityMTL
from data_loader import ToxicityDataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subgroup", type=str, default="black")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_name", type=str, default="feature_space_viz.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "microsoft/deberta-v3-base"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = DebertaToxicityMTL(MODEL_PATH).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 加载验证集
    VAL_FILE = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    df = pd.read_parquet(VAL_FILE)
    
    # 筛选子特定子群体
    sub_df = df[df[args.subgroup] >= 0.5].copy()
    print(f"Subgroup '{args.subgroup}' has {len(sub_df)} samples.")
    
    # 为了可视化效果更好，我们可以平衡一下 toxic 和 non-toxic 的比例
    toxic_df = sub_df[sub_df['target'] >= 0.5]
    nontoxic_df = sub_df[sub_df['target'] < 0.5]
    
    n_per_class = args.n_samples // 2
    toxic_sample = toxic_df.sample(min(len(toxic_df), n_per_class))
    nontoxic_sample = nontoxic_df.sample(min(len(nontoxic_df), n_per_class))
    
    plot_df = pd.concat([toxic_sample, nontoxic_sample]).sample(frac=1.0).reset_index(drop=True)
    print(f"Visualizing {len(plot_df)} samples total.")
    
    # 提取特征
    ds = ToxicityDataset(plot_df, tokenizer, max_len=256)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            
            # 提取 project 层的特征 (z)
            feat = outputs['features'].cpu().numpy()
            features.append(feat)
            labels.extend((batch['y_tox'] >= 0.5).numpy().astype(int))
            
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    
    # t-SNE 降维
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    feat_2d = tsne.fit_transform(features)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    sns.set_style("white")
    
    # 自定义颜色
    # Toxic: Coral, Non-Toxic: MediumSeaGreen (参照用户截图)
    colors = ["#3CB371", "#FF7F50"] # Green for 0, Coral for 1
    
    for i, label_name in enumerate(["Non-Toxic", "Toxic"]):
        idx = (labels == i)
        plt.scatter(feat_2d[idx, 0], feat_2d[idx, 1], c=colors[i], label=label_name, alpha=0.7, edgecolors='w', s=60)
    
    plt.title(f"Feature Space Visualization (Identity: '{args.subgroup}')", fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True, bottom=True)
    
    output_path = os.path.join(BASE_DIR, "src_result", args.output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
