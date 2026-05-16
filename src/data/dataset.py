"""
反事实公平性训练的数据加载模块。

CausalFairDataset 将每条原始文本与其 LLM 生成的反事实配对，
并提供统一的训练接口。
"""
from typing import Dict, List, Optional, Callable
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CausalFairDataset(Dataset):
    """包含原始文本及其反事实配对的数据集。

    对于没有反事实的样本，has_cf=0，cf 字段为零张量。
    这允许使用单一统一的 DataLoader 处理有/无反事实的样本。

    df 必须包含的列：
        - text: 原始文本
        - binary_label: 0/1 标签

    cf_df 必须包含的列：
        - original_text: 必须与 df['text'] 匹配
        - cf_text: 反事实版本
        - source_group: 被替换的身份群体（可选）
        - target_group: 引入的身份群体（可选）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cf_df: Optional[pd.DataFrame],
        tokenizer,
        max_len: int = 128,
        cf_sample_ratio: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df["text"].values
        self.labels = df["binary_label"].values

        # 构建 文本 → 反事实文本列表 映射
        self.cf_map: Dict[str, List[str]] = {}
        if cf_df is not None and len(cf_df) > 0:
            for _, row in cf_df.iterrows():
                orig = str(row["original_text"])
                if orig not in self.cf_map:
                    self.cf_map[orig] = []
                self.cf_map[orig].append(row["cf_text"])

        # 仅筛选有反事实的样本（用于 CLP 训练）
        self._cf_indices = [
            i for i, t in enumerate(self.texts) if t in self.cf_map
        ]
        self._all_indices = list(range(len(self.texts)))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 对原始文本进行分词
        orig_enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        item = {
            "orig_input_ids": torch.tensor(orig_enc["input_ids"], dtype=torch.long),
            "orig_attention_mask": torch.tensor(orig_enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

        # 如果有反事实文本，则进行分词
        if text in self.cf_map and len(self.cf_map[text]) > 0:
            cf_text = random.choice(self.cf_map[text])
            cf_enc = self.tokenizer.encode_plus(
                str(cf_text),
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
            item["cf_input_ids"] = torch.tensor(cf_enc["input_ids"], dtype=torch.long)
            item["cf_attention_mask"] = torch.tensor(cf_enc["attention_mask"], dtype=torch.long)
            item["has_cf"] = torch.tensor(1, dtype=torch.long)
        else:
            # 占位张量（训练时将被掩码屏蔽）
            item["cf_input_ids"] = torch.zeros(self.max_len, dtype=torch.long)
            item["cf_attention_mask"] = torch.zeros(self.max_len, dtype=torch.long)
            item["has_cf"] = torch.tensor(0, dtype=torch.long)

        return item


def causal_fair_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """自定义 collate 函数：堆叠所有字段，cf 字段单独处理。

    所有原始样本均被堆叠。cf 字段仅包含 has_cf=1 的样本。
    """
    keys = [
        "orig_input_ids",
        "orig_attention_mask",
        "label",
        "has_cf",
    ]
    result = {k: torch.stack([b[k] for b in batch]) for k in keys}

    cf_items = [b for b in batch if b["has_cf"].item() == 1]
    if cf_items:
        result["cf_input_ids"] = torch.stack([b["cf_input_ids"] for b in cf_items])
        result["cf_attention_mask"] = torch.stack([b["cf_attention_mask"] for b in cf_items])
        result["cf_indices"] = torch.tensor(
            [i for i, b in enumerate(batch) if b["has_cf"].item() == 1],
            dtype=torch.long,
        )

    return result


def get_causal_fair_loader(
    df: pd.DataFrame,
    cf_df: Optional[pd.DataFrame],
    tokenizer,
    batch_size: int = 16,
    max_len: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """构建用于反事实公平性训练的 DataLoader。

    Args:
        df: 原始数据集 DataFrame。
        cf_df: 反事实对 DataFrame（或 None）。
        tokenizer: HuggingFace 分词器。
        batch_size: 批次大小。
        max_len: 最大序列长度。
        shuffle: 是否打乱数据。
        num_workers: DataLoader 工作进程数。
        pin_memory: 为 GPU 固定内存。

    Returns:
        使用自定义 collate 函数的 DataLoader。
    """
    dataset = CausalFairDataset(df, cf_df, tokenizer, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=causal_fair_collate_fn,
    )