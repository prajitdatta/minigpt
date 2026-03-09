"""
Optimizer Configuration
=======================

Implements AdamW with proper weight decay separation:
weight decay is NOT applied to bias terms, LayerNorm,
or embedding parameters.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

from typing import Tuple

import torch
import torch.nn as nn


def configure_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Configure AdamW optimizer with proper weight decay handling.

    Parameters that should NOT have weight decay:
    - All bias parameters
    - LayerNorm weight and bias
    - Embedding parameters

    Args:
        model: The model to optimize.
        learning_rate: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Adam epsilon for numerical stability.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    no_decay_names = set()
    decay_names = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't decay biases, LayerNorm, or embeddings
        if (
            param.dim() <= 1
            or "bias" in name
            or "ln" in name
            or "layernorm" in name.lower()
            or "embedding" in name
        ):
            no_decay_params.append(param)
            no_decay_names.add(name)
        else:
            decay_params.append(param)
            decay_names.add(name)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Log parameter counts
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(
        f"Optimizer: {n_decay:,} params with decay, "
        f"{n_no_decay:,} params without decay"
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),
    )

    return optimizer
