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
from src.counterfactual.generator_llm import (
    ZhipuGenerator,
    ZhipuGeneratorPool,
    OpenAICompatGenerator,
    generate_counterfactuals_for_dataset,
)
from src.counterfactual.generator_swap import (
    generate_swap_counterfactuals,
    batch_generate_swap,
)
from src.counterfactual.validator import (
    validate_counterfactual,
    validate_and_filter,
)

__all__ = [
    "PROMPT_FIXED_MAPPING",
    "PROMPT_FREE_FORM",
    "IDENTITY_SWAP_PAIRS",
    "GROUP_KEYWORDS",
    "CounterfactualPair",
    # LLM Generators
    "ZhipuGenerator",
    "ZhipuGeneratorPool",
    "OpenAICompatGenerator",
    "generate_counterfactuals_for_dataset",
    # Swap Generators
    "generate_swap_counterfactuals",
    "batch_generate_swap",
    # Validators
    "validate_counterfactual",
    "validate_and_filter",
]