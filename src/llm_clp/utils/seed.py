"""
统一随机种子管理模块。

同时设置 Python、NumPy、PyTorch 和 CUDA 的随机种子。
"""
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """为可复现性设置所有随机种子。

    Args:
        seed: 随机种子（如 42）。
        deterministic: 若为 True，同时启用 CuDNN 确定性模式。
                      该选项可能会降低训练速度，但能保证结果完全可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 在 Ampere+ 架构 GPU 上启用 TF32 确定性模式
        if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False