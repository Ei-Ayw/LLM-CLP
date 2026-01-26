import torch
import numpy as np

class EarlyStopping:
    """
    早停机制 (Early Stopping)：
    当验证集损失在指定的 patience 轮数内不再下降时，提前结束训练。
    """
    def __init__(self, patience=3, min_delta=0, verbose=True):
        """
        Args:
            patience (int): 等待轮数
            min_delta (float): 最小变化阈值
            verbose (bool): 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Loss improved. Resetting counter.")
        return self.early_stop
