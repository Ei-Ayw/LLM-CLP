class EarlyStopping:
    """早停机制：当监控指标不再改善时提前终止训练。"""

    def __init__(self, patience=3, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None

    def __call__(self, value):
        if self.best is None:
            self.best = value
            return False

        improved = value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta
        if improved:
            self.best = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
