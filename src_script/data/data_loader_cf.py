"""
=============================================================================
数据加载器: 反事实配对数据集
用于训练时同时加载原始样本和对应的反事实样本
=============================================================================
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CausalFairDataset(Dataset):
    """
    反事实配对数据集

    每个样本返回:
      - 原始文本的 tokenized inputs
      - 反事实文本的 tokenized inputs (如果有)
      - 标签
      - 是否有反事实配对 (has_cf)
    """

    def __init__(self, df, cf_df, tokenizer, max_len=128, num_classes=2):
        """
        Args:
            df: 原始数据 DataFrame, 需要 text, binary_label 列
            cf_df: 反事实数据 DataFrame, 需要 post_id/original_text, cf_text 列
                   可以为 None (baseline 模式)
            tokenizer: HuggingFace tokenizer
            max_len: 最大序列长度
            num_classes: 类别数
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

        self.texts = df['text'].values
        self.labels = df['binary_label'].values

        # 建立原始文本到反事实的映射
        self.cf_map = {}
        if cf_df is not None and len(cf_df) > 0:
            for _, row in cf_df.iterrows():
                orig = row['original_text']
                if orig not in self.cf_map:
                    self.cf_map[orig] = []
                self.cf_map[orig].append(row['cf_text'])

    def __len__(self):
        return len(self.texts)

    def _tokenize(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
        }

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 原始样本
        orig_inputs = self._tokenize(text)

        # 反事实样本
        cf_list = self.cf_map.get(text, [])
        has_cf = len(cf_list) > 0

        if has_cf:
            # 随机选一个反事实
            cf_text = cf_list[np.random.randint(len(cf_list))]
            cf_inputs = self._tokenize(cf_text)
        else:
            # 没有反事实时，用原文本自身 (CLP loss 为 0)
            cf_inputs = self._tokenize(text)

        return {
            'orig_input_ids': orig_inputs['input_ids'],
            'orig_attention_mask': orig_inputs['attention_mask'],
            'cf_input_ids': cf_inputs['input_ids'],
            'cf_attention_mask': cf_inputs['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long),
            'has_cf': torch.tensor(1 if has_cf else 0, dtype=torch.long),
        }


def get_causal_fair_loader(df, cf_df, tokenizer, batch_size=16, max_len=128,
                           shuffle=True, num_workers=2):
    """快捷函数: 创建 DataLoader"""
    dataset = CausalFairDataset(df, cf_df, tokenizer, max_len=max_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )
