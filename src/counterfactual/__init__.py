"""
反事实生成与验证工具集。
"""
from src.counterfactual.prompts import (
    PROMPT_FIXED_MAPPING,
    PROMPT_FREE_FORM,
    IDENTITY_SWAP_PAIRS,
    GROUP_KEYWORDS,
)
from src.counterfactual.schema import CounterfactualPair

__all__ = [
    "PROMPT_FIXED_MAPPING",
    "PROMPT_FREE_FORM",
    "IDENTITY_SWAP_PAIRS",
    "GROUP_KEYWORDS",
    "CounterfactualPair",
]