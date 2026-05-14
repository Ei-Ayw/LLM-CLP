"""
LLM-CLP: Counterfactual Logit Pairing with LLM-Generated Counterfactuals

A research library for identity-fair toxicity classification using counterfactual
data augmentation and logit pairing regularization.

Dependencies: torch, transformers, pandas, scikit-learn, tqdm
"""

__version__ = "1.0.0"
__author__ = "LLM-CLP Team"

__all__ = [
    "CounterfactualLogitPairing",
    "CounterfactualSupConLoss",
    "compute_cfr",
    "compute_ctfg",
    "compute_fped_fned",
]


def __getattr__(name):
    """Lazy import to avoid importing heavy dependencies at package load."""
    if name == "CounterfactualLogitPairing" or name == "CounterfactualSupConLoss":
        from src.llm_clp.models.losses import (
            CounterfactualLogitPairing,
            CounterfactualSupConLoss,
        )
        return CounterfactualLogitPairing if name == "CounterfactualLogitPairing" else CounterfactualSupConLoss
    if name in ("compute_cfr", "compute_ctfg", "compute_fped_fned"):
        from src.llm_clp.evaluation.metrics import (
            compute_cfr,
            compute_ctfg,
            compute_fped_fned,
        )
        return {"compute_cfr": compute_cfr, "compute_ctfg": compute_ctfg, "compute_fped_fned": compute_fped_fned}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")