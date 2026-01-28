import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# =============================================================================
# ### 数据类：ToxicityDataset ###
# 设计说明：
# 该类负责将预处理后的 Parquet 数据转换为 PyTorch 可用的张量格式。
# 除了主任务（毒性分类）标签外，还会加载细分类别标签和身份属性标签，
# 以配合多任务学习（MTL）架构。
# =============================================================================
class ToxicityDataset(Dataset):
    """
    通用毒性分类数据集类，封装了文本编码与多任务标签处理。
    """
    def __init__(self, df, tokenizer, max_len=256, augment=False):
        self.tokenizer = tokenizer
        self.texts = df['comment_text'].values
        # 主任务标签：二分类毒性
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

    def simple_aug(self, text, p=0.1):
        """ 简单的随机删除增强 (Random Deletion) """
        words = text.split()
        if len(words) <= 1: return text
        # 随机保留单词，每个单词有 1-p 的概率保留
        new_words = [w for w in words if np.random.rand() > p]
        # 如果全部删完了，则至少保留一个
        if not new_words: return words[np.random.randint(0, len(words))]
        return " ".join(new_words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        y_tox_val = self.y_tox[idx]

        # 数据增强策略: 仅对少数类 (Toxic) 及其子类样本进行随机扰动
        # 这有助于增加少数类样本的多样性，缓解类别不平衡
        if self.augment and y_tox_val >= 0.5:
             # 50% 概率触发增强，避免改变过大
            if np.random.rand() < 0.5:
                text = self.simple_aug(text, p=0.1)
        
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
            'y_tox': torch.tensor(y_tox_val, dtype=torch.float),
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
    根据用户要求，统一对齐所有实验的训练数据量。
    """
    if len(df) <= n_samples:
        return df
    return df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    # 单元测试
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", local_files_only=True)
    mock_df = pd.DataFrame({
        'comment_text': ["Example sentence"] * 10,
        'y_tox': [0.0] * 10,
        'severe_toxicity': [0.0] * 10, 'obscene': [0.0] * 10, 'threat': [0.0] * 10,
        'insult': [0.0] * 10, 'identity_attack': [0.0] * 10, 'sexual_explicit': [0.0] * 10,
        'male': [0.0] * 10, 'female': [0.0] * 10, 'black': [0.0] * 10, 'white': [0.0] * 10,
        'muslim': [0.0] * 10, 'jewish': [0.0] * 10, 'christian': [0.0] * 10,
        'homosexual_gay_or_lesbian': [0.0] * 10, 'psychiatric_or_mental_illness': [0.0] * 10,
        'has_identity': [0] * 10
    })
    
    loader = get_dataloader(mock_df, tokenizer, batch_size=2)
    batch = next(iter(loader))
    print(">>> 数据加载器测试成功，Batch 字段如下：")
    print(batch.keys())
