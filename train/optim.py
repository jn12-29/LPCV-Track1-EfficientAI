from __future__ import annotations

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    total_steps = max(total_steps, 1)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (current_step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
