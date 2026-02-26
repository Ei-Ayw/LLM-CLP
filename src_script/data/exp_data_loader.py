import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# =============================================================================
# ### 数据类：ToxicityDataset ###
# 已移除所有在线数据增强 (safe_aug/simple_aug), 原因:
#   - 随机交换/删词会破坏毒性文本关键语义, 引入标签噪声
#   - DeBERTa 预训练已具备足够泛化能力, dropout=0.2 已提供正则化
# =============================================================================
class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, augment=False):
        self.tokenizer = tokenizer
        self.texts = df['comment_text'].values
        # 优先使用 soft label (连续值 [0,1])，回退到二值标签
        if 'y_tox_soft' in df.columns:
            self.y_tox = df['y_tox_soft'].values
        else:
            self.y_tox = df['y_tox'].values

        self.subtype_cols = [
            'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'
        ]
        self.identity_cols = [
            'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian',
            'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
        ]

        self.y_sub = df[self.subtype_cols].values
        self.y_id = df[self.identity_cols].values
        self.has_id = df['has_identity'].values
        self.max_len = max_len
        # augment 参数保留但不再使用, 仅为兼容旧脚本调用不报错

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'y_tox': torch.tensor(self.y_tox[idx], dtype=torch.float),
            'y_sub': torch.tensor(self.y_sub[idx], dtype=torch.float),
            'y_id': torch.tensor(self.y_id[idx], dtype=torch.float),
            'has_id': torch.tensor(self.has_id[idx], dtype=torch.long)
        }

def get_dataloader(df, tokenizer, batch_size=16, max_len=256, shuffle=True, num_workers=4):
    dataset = ToxicityDataset(df, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def sample_aligned_data(df, n_samples=200000, seed=42):
    if len(df) <= n_samples:
        n_toxic = (df['y_tox'] >= 0.5).sum()
        n_normal = len(df) - n_toxic
        print(f"  [Data Balance] 数据已预处理: {len(df)} 条 (有毒: {n_toxic}, 正常: {n_normal}) - 跳过采样")
        return df

    toxic_df = df[df['y_tox'] >= 0.5]
    normal_df = df[df['y_tox'] < 0.5]
    print(f"  [Data Balance] 原始数据: {len(df)} 条 (有毒: {len(toxic_df)}, 正常: {len(normal_df)})")

    n_per_class = n_samples // 2
    sampled_toxic = toxic_df.sample(n=min(n_per_class, len(toxic_df)), random_state=seed)
    sampled_normal = normal_df.sample(n=min(n_per_class, len(normal_df)), random_state=seed)

    result = pd.concat([sampled_toxic, sampled_normal], ignore_index=True)
    print(f"  [Data Balance] 平衡采样完成: {len(result)} 条 (有毒: {len(sampled_toxic)}, 正常: {len(sampled_normal)})")
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)
