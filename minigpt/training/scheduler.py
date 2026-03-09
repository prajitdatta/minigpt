"""
Learning Rate Schedulers
========================

Cosine annealing with warmup, linear warmup, and custom schedules.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 100000,
    warmup_steps: int = 2000,
    min_lr: float = 3e-5,
) -> Optional[LambdaLR]:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant').
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        min_lr: Minimum learning rate (for cosine annealing).

    Returns:
        LambdaLR scheduler instance, or None for constant.
    """
    base_lr = optimizer.param_groups[0]["lr"]

    if scheduler_type == "cosine":
        def lr_lambda(step):
            # Linear warmup
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            # Cosine annealing
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cosine_decay
            return lr / base_lr

    elif scheduler_type == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_lr / base_lr, 1.0 - progress)

    elif scheduler_type == "constant":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineScheduler:
    """
    Standalone cosine annealing scheduler with warmup.

    Can be used without PyTorch's LR scheduler infrastructure.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        """Update the learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        step = self.current_step

        if step < self.warmup_steps:
            return self.base_lr * step / max(self.warmup_steps, 1)

        if step >= self.total_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine
