"""
Data loading for counterfactual fairness training.

CausalFairDataset pairs each original text with its LLM-generated counterfactual
and exposes a unified training interface.
"""
from typing import Dict, List, Optional, Callable
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CausalFairDataset(Dataset):
    """Dataset with original texts and their counterfactual pairs.

    For samples without counterfactuals, has_cf=0 and cf fields are zero-tensors.
    This allows a single unified DataLoader for both CF and non-CF samples.

    Columns required in df:
        - text: original text
        - binary_label: 0/1 label

    Columns required in cf_df:
        - original_text: must match df['text']
        - cf_text: counterfactual version
        - source_group: identity group replaced (optional)
        - target_group: identity group substituted (optional)
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

        # Build text → list of CF texts mapping
        self.cf_map: Dict[str, List[str]] = {}
        if cf_df is not None and len(cf_df) > 0:
            for _, row in cf_df.iterrows():
                orig = str(row["original_text"])
                if orig not in self.cf_map:
                    self.cf_map[orig] = []
                self.cf_map[orig].append(row["cf_text"])

        # Filter to only samples with CF (for CLP training)
        self._cf_indices = [
            i for i, t in enumerate(self.texts) if t in self.cf_map
        ]
        self._all_indices = list(range(len(self.texts)))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize original
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

        # Tokenize CF if available
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
            # Placeholder tensors (will be masked in training)
            item["cf_input_ids"] = torch.zeros(self.max_len, dtype=torch.long)
            item["cf_attention_mask"] = torch.zeros(self.max_len, dtype=torch.long)
            item["has_cf"] = torch.tensor(0, dtype=torch.long)

        return item


def causal_fair_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate: stacks all fields, keeps CF fields separate.

    All original samples are stacked. CF fields only contain samples with has_cf=1.
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
    """Build a DataLoader for counterfactual fairness training.

    Args:
        df: Original dataset DataFrame.
        cf_df: Counterfactual pairs DataFrame (or None).
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size.
        max_len: Max sequence length.
        shuffle: Whether to shuffle.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU.

    Returns:
        DataLoader with custom collate.
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