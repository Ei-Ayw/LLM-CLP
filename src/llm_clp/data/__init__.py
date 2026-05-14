"""
Data loading utilities for LLM-CLP.

Provides:
- CausalFairDataset: pairs original texts with LLM-generated counterfactuals
- get_causal_fair_loader: builds DataLoader with proper collate
"""
from src.llm_clp.data.dataset import CausalFairDataset, get_causal_fair_loader

__all__ = ["CausalFairDataset", "get_causal_fair_loader"]