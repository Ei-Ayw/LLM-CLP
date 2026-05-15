"""
LLM-CLP: 使用 LLM 生成的反事实进行对数配对

一个用于身份公平毒性分类的研究库，使用反事实数据增强和对数配对正则化。

依赖项：torch, transformers, pandas, scikit-learn, tqdm
"""

__version__ = "1.0.0"
__author__ = "LLM-CLP Team"

__all__ = [
    "CounterfactualLogitPairing",
    "CounterfactualSupConLoss",
    "compute_cfr",
    "compute_ctfg",
    "compute_fped_fned",
    "_losses",
    "_metrics",
]


def __getattr__(name):
    """延迟导入，避免在包加载时引入大型依赖。"""
    if name == "CounterfactualLogitPairing" or name == "CounterfactualSupConLoss":
        from .models.losses import (
            CounterfactualLogitPairing,
            CounterfactualSupConLoss,
        )
        return CounterfactualLogitPairing if name == "CounterfactualLogitPairing" else CounterfactualSupConLoss
    if name in ("compute_cfr", "compute_ctfg", "compute_fped_fned"):
        from .eval.metrics import (
            compute_cfr,
            compute_ctfg,
            compute_fped_fned,
        )
        return {"compute_cfr": compute_cfr, "compute_ctfg": compute_ctfg, "compute_fped_fned": compute_fped_fned}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 旧导入兼容层
from .models import losses as _losses
from .eval import metrics as _metrics

__all__.extend([
    "_losses",
    "_metrics",
])