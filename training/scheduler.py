from __future__ import annotations

import math
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    import torch

    from config import TrainingConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> LambdaLR:
    """Linear warmup followed by cosine decay to min_lr_ratio * lr."""
    warmup = config.warmup_steps
    max_steps = config.max_steps
    min_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
