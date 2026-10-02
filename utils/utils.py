import os
import random
import numpy as np
import torch

class EarlyStopping:
    """
    早停机制，支持按 val_loss 或 val_accuracy 监视。

    Parameters
    ----------
    patience : int
        连续多少轮不改善后触发早停。
    monitor : str
        'loss'（默认）监视验证损失最小化；
        'accuracy' 监视验证准确率最大化。
    """
    def __init__(self, patience=10, monitor='loss'):
        self.patience = patience
        self.monitor = monitor
        self.counter = 0
        if monitor == 'loss':
            self.best_value = float('inf')
            self._improved = lambda curr, best: curr < best
        elif monitor == 'accuracy':
            self.best_value = float('-inf')
            self._improved = lambda curr, best: curr > best
        else:
            raise ValueError(f"monitor must be 'loss' or 'accuracy', got {monitor}")
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, val_loss, val_acc):
        """
        epoch : int   当前 epoch 编号（从 1 开始）
        val_loss : float
        val_acc : float
        """
        current = val_loss if self.monitor == 'loss' else val_acc
        if self._improved(current, self.best_value):
            self.best_value = current
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
