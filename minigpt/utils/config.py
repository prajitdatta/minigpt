"""
Configuration Management
========================

Defines model configurations as dataclasses with
pre-built presets for GPT-2 family models.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Complete model configuration.

    This is the central configuration object that controls
    all aspects of the MiniGPT model architecture.
    """

    # Architecture
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    block_size: int = 1024

    # Regularization
    dropout: float = 0.1
    embd_dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1

    # Options
    bias: bool = True
    activation: str = "gelu"
    pos_type: str = "learned"
    flash_attention: bool = True
    weight_tying: bool = True

    # Name
    name: str = "minigpt-custom"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 50
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    # Scheduling
    scheduler: str = "cosine"
    warmup_steps: int = 2000
    min_lr: float = 3e-5

    # Precision
    mixed_precision: bool = True
    compile_model: bool = False

    # Data
    num_workers: int = 4
    pin_memory: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Distributed
    distributed: Optional[str] = None  # "ddp" or "fsdp"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None


# ============================================================
# Pre-built model configurations (GPT-2 family)
# ============================================================


def GPT2SmallConfig() -> ModelConfig:
    """GPT-2 Small: 124M parameters."""
    return ModelConfig(
        vocab_size=50257,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        block_size=1024,
        dropout=0.1,
        name="gpt2-small",
    )


def GPT2MediumConfig() -> ModelConfig:
    """GPT-2 Medium: 355M parameters."""
    return ModelConfig(
        vocab_size=50257,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
        block_size=1024,
        dropout=0.1,
        name="gpt2-medium",
    )


def GPT2LargeConfig() -> ModelConfig:
    """GPT-2 Large: 774M parameters."""
    return ModelConfig(
        vocab_size=50257,
        d_model=1280,
        n_heads=20,
        n_layers=36,
        d_ff=5120,
        block_size=1024,
        dropout=0.1,
        name="gpt2-large",
    )


def GPT2XLConfig() -> ModelConfig:
    """GPT-2 XL: 1.5B parameters."""
    return ModelConfig(
        vocab_size=50257,
        d_model=1600,
        n_heads=25,
        n_layers=48,
        d_ff=6400,
        block_size=1024,
        dropout=0.1,
        name="gpt2-xl",
    )


# Config registry for CLI lookups
CONFIG_REGISTRY = {
    "gpt2-small": GPT2SmallConfig,
    "gpt2-medium": GPT2MediumConfig,
    "gpt2-large": GPT2LargeConfig,
    "gpt2-xl": GPT2XLConfig,
}


def get_config(name: str) -> ModelConfig:
    """Get a model config by name."""
    if name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return CONFIG_REGISTRY[name]()
