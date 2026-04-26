from __future__ import annotations

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    init_lr: float = 0.0,
    min_lr: float = 0.0,
) -> LambdaLR:
    total_steps = max(total_steps, 1)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    init_scale = init_lr / peak_lr if peak_lr > 0 else 0.0
    min_scale = min_lr / peak_lr if peak_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            progress = float(current_step) / float(warmup_steps)
            return init_scale + (1.0 - init_scale) * progress
        if total_steps <= warmup_steps:
            return 1.0
        progress = (current_step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    return LambdaLR(optimizer, lr_lambda)
