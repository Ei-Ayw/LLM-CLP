import torch
import numpy as np

# =============================================================================
# 全局硬件加速配置 (针对 A10/A100/3090+ 显卡)
# =============================================================================
if torch.cuda.is_available():
    # 开启 TF32 (TensorFloat-32) 加速矩阵运算
    # 这通常能带来 2-5 倍的性能提升，且精度损失微乎其微
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[Hardware] TF32 acceleration enabled for Ampere+ GPUs.")

class EarlyStopping:
    """
    早停机制 (Early Stopping)：
    当验证集指标在指定的 patience 轮数内不再改善时，提前结束训练。
    支持最小化模式（如 loss）和最大化模式（如 F1）。
    """
    def __init__(self, patience=3, min_delta=0, verbose=True, mode="min"):
        """
        Args:
            patience (int): 等待轮数
            min_delta (float): 最小变化阈值
            verbose (bool): 是否打印信息
            mode (str): "min" 最小化指标（损失），"max" 最大化指标（ F1/AUC）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, val_value):
        if self.best_value is None:
            self.best_value = val_value
        elif self.mode == "min" and val_value > self.best_value - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == "max" and val_value < self.best_value + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_value = val_value
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Improved. Resetting counter.")
        return self.early_stop
