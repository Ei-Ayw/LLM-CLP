"""
反事实生成与验证工具集。
"""
from src.llm_clp.counterfactual.prompts import (
    PROMPT_FIXED_MAPPING,
    PROMPT_FREE_FORM,
    IDENTITY_SWAP_PAIRS,
    GROUP_KEYWORDS,
)
from src.llm_clp.counterfactual.schema import CounterfactualPair

__all__ = [
    "PROMPT_FIXED_MAPPING",
    "PROMPT_FREE_FORM",
    "IDENTITY_SWAP_PAIRS",
    "GROUP_KEYWORDS",
    "CounterfactualPair",
]