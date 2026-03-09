"""
Pre-built Model Configurations
===============================

Author: Prajit Datta (https://github.com/prajitdatta)
"""

from minigpt.utils.config import (
    ModelConfig,
    TrainingConfig,
    GPT2SmallConfig,
    GPT2MediumConfig,
    GPT2LargeConfig,
    GPT2XLConfig,
    get_config,
    CONFIG_REGISTRY,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "GPT2SmallConfig",
    "GPT2MediumConfig",
    "GPT2LargeConfig",
    "GPT2XLConfig",
    "get_config",
    "CONFIG_REGISTRY",
]
