"""
LLM-CLP 数据加载工具集。

提供：
- CausalFairDataset：将原始文本与 LLM 生成的反事实配对
- get_causal_fair_loader：构建带正确 collate 函数的 DataLoader
"""
from src.llm_clp.data.dataset import CausalFairDataset, get_causal_fair_loader

__all__ = ["CausalFairDataset", "get_causal_fair_loader"]