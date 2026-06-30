from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """AdamW with separate weight decay groups.

    Group A (weight matrices, ndim >= 2, non-embedding): weight_decay = config.weight_decay
    Group B (norms, biases, embeddings): weight_decay = 0.0
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or "embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        groups,
        lr=config.lr,
        betas=config.betas,
        eps=1e-8,
    )
