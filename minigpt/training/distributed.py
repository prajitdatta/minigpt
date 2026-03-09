"""
Distributed Training Utilities
===============================

Helpers for DDP and FSDP multi-GPU / multi-node training.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(
    backend: str = "nccl",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """
    Initialize distributed training.

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).
        rank: Process rank (auto-detected from env if None).
        world_size: Total number of processes.
    """
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print(f"Distributed training initialized: rank {rank}/{world_size}")


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
) -> DDP:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: The model to wrap.
        device_id: GPU device ID for this process.
        find_unused_parameters: Enable if some params don't get gradients.

    Returns:
        DDP-wrapped model.
    """
    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_model_fsdp(model: torch.nn.Module):
    """
    Wrap model with Fully Sharded Data Parallel (FSDP).

    Requires PyTorch 2.0+.

    Args:
        model: The model to wrap.

    Returns:
        FSDP-wrapped model.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    return FSDP(model, mixed_precision=mp_policy)


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
