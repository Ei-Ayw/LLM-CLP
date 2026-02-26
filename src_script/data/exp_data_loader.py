import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# =============================================================================
# ### 数据类：ToxicityDataset ###
# 改进点:
# 1. 支持 soft label (保留原始 target 连续值 [0,1])
# 2. 数据增强修复: 不再只针对 toxic 样本，对所有样本低概率增强
# 3. 增强方式更安全: 随机交换相邻词 (不删词，避免破坏语义)
# =============================================================================
class ToxicityDataset(Dataset):
    """
    通用毒性分类数据集类，封装了文本编码与多任务标签处理。
    """
    def __init__(self, df, tokenizer, max_len=256, augment=False):
        self.tokenizer = tokenizer
        self.texts = df['comment_text'].values
        # 主任务标签：优先使用 soft label (连续值 [0,1])，回退到二值标签
        if 'y_tox_soft' in df.columns:
            self.y_tox = df['y_tox_soft'].values
        else:
            self.y_tox = df['y_tox'].values

        # 细分任务标签 (Subtype)
        self.subtype_cols = [
            'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'
        ]
        # 身份相关标签 (Identity)
        self.identity_cols = [
            'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian',
            'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
        ]

        self.y_sub = df[self.subtype_cols].values
        self.y_id = df[self.identity_cols].values
        # 标识是否包含身份提及
        self.has_id = df['has_identity'].values
        self.max_len = max_len
        self.augment = augment

    def safe_aug(self, text, p=0.1):
        """
        安全的数据增强: 随机交换相邻词对 (Random Swap)
        相比随机删词，不会破坏关键语义信息
        """
        words = text.split()
        if len(words) <= 2:
            return text
        n_swaps = max(1, int(len(words) * p))
        new_words = words.copy()
        for _ in range(n_swaps):
            idx = np.random.randint(0, len(new_words) - 1)
            new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
        return " ".join(new_words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # 数据增强: 对所有样本低概率增强 (20% 概率)
        # 不再仅针对 toxic 样本 (数据已1:1平衡)
        if self.augment and np.random.rand() < 0.2:
            text = self.safe_aug(text, p=0.1)

        # 使用 Transformers 的 encode_plus 进行标准化编码
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
    """
    创建数据加载器。已将文件读取逻辑移出，以便更好地控制数据的采样与对齐。
    """
    dataset = ToxicityDataset(df, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def sample_aligned_data(df, n_samples=200000, seed=42):
    """
    类别平衡采样策略：
    - 如果数据已经 <= 目标大小，直接返回（避免重复采样）
    - 否则进行 50:50 平衡采样
    """
    # 如果数据已经预处理过了（小于等于目标），直接返回
    if len(df) <= n_samples:
        n_toxic = (df['y_tox'] >= 0.5).sum()
        n_normal = len(df) - n_toxic
        print(f"  [Data Balance] 数据已预处理: {len(df)} 条 (有毒: {n_toxic}, 正常: {n_normal}) - 跳过采样")
        return df

    # 以下是全量数据的平衡采样逻辑
    toxic_df = df[df['y_tox'] >= 0.5]
    normal_df = df[df['y_tox'] < 0.5]

    n_toxic = len(toxic_df)
    n_normal = len(normal_df)

    print(f"  [Data Balance] 原始数据: {len(df)} 条 (有毒: {n_toxic}, 正常: {n_normal})")

    # 50:50 平衡采样
    n_per_class = n_samples // 2

    sampled_toxic = toxic_df.sample(n=min(n_per_class, n_toxic), random_state=seed)
    sampled_normal = normal_df.sample(n=min(n_per_class, n_normal), random_state=seed)

    result = pd.concat([sampled_toxic, sampled_normal], ignore_index=True)
    print(f"  [Data Balance] 平衡采样完成: {len(result)} 条 (有毒: {len(sampled_toxic)}, 正常: {len(sampled_normal)})")

    # 打乱顺序
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)
