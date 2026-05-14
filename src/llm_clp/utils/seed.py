"""
Unified random seed management.

Sets Python, NumPy, PyTorch, and CUDA seeds simultaneously.
"""
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed (e.g. 42).
        deterministic: If True, also enable CuDNN deterministic mode.
                      This may slow down training but ensures exact reproducibility.
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
        # Enable TF32 deterministic mode on Ampere+ GPUs
        if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False