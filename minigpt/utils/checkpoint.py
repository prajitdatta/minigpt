"""
Checkpoint Management
=====================

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import os
import glob
from typing import Optional, Dict, Any

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    **extra,
):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "global_step": step,
        "loss": loss,
        "config": model.config if hasattr(model, "config") else None,
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model=None,
    optimizer=None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)
