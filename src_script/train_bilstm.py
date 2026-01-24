import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
from collections import Counter
import pickle

# =============================================================================
# ### 训练脚本：train_bilstm.py ###
# 设计说明：
# 本脚本用于训练标准的双向 LSTM 模型。
# 用于评估长文本序列建模在毒性分类任务中的必要性，是对比实验中的重要一环。
# =============================================================================

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src_model"))
sys.path.append(os.path.join(BASE_DIR, "src_script"))

from model_classic_deep import BiLSTM
from data_loader import sample_aligned_data

# --- 词表构建（与 CNN 保持一致以保证公平对比） ---
class SimpleVocab:
    def __init__(self, texts, max_size=20000):
        counter = Counter()
        for text in tqdm(texts, desc="Building Vocab"):
            counter.update(str(text).lower().split())
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for word, count in counter.most_common(max_size - 2):
            self.stoi[word] = len(self.stoi)

    def encode(self, text, max_len=256):
        tokens = str(text).lower().split()
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

def main():
    parser = argparse.ArgumentParser(description="BiLSTM Baseline Training")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_name = f"BiLSTM_Sample{args.sample_size}_{timestamp}.pth"
    save_path = os.path.join(BASE_DIR, "src_result", save_name)

    print(f"\n>>> 启动 BiLSTM 实验: {save_name}")

    train_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "train_processed.parquet"))
    val_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "val_processed.parquet"))
    train_df = sample_aligned_data(train_df, n_samples=args.sample_size, seed=args.seed)
    
    vocab = SimpleVocab(pd.concat([train_df['comment_text'], val_df['comment_text']]))
    vocab_size = len(vocab.stoi)
    
    def prepare_data(df):
        texts = [vocab.encode(t, args.max_len) for t in tqdm(df['comment_text'], desc="Encoding")]
        labels = df['y_tox'].values
        return torch.tensor(texts), torch.tensor(labels, dtype=torch.float)

    train_ids, train_labels = prepare_data(train_df)
    val_ids, val_labels = prepare_data(val_df)
    train_loader = DataLoader(list(zip(train_ids, train_labels)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_ids, val_labels)), batch_size=args.batch_size, shuffle=False)

    model = BiLSTM(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for ids, targets in pbar:
            ids, targets = ids.to(device), targets.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            out = model(ids)
            loss = criterion(out['logits_tox'], targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/len(pbar):.4f}")
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for ids, targets in val_loader:
                ids, targets = ids.to(device), targets.to(device).unsqueeze(-1)
                out = model(ids)
                v_loss += criterion(out['logits_tox'], targets).item()
        val_loss = v_loss / len(val_loader)
        
        print(f"  Result -> Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 同时保存权重和词表
            torch.save(model.state_dict(), save_path)
            vocab_path = save_path.replace(".pth", "_vocab.pkl")
            with open(vocab_path, 'wb') as f:
                pickle.dump(vocab, f)
            print(f"  [Save] 更优权重及词表已保存: {save_path}")

if __name__ == "__main__":
    main()
