"""
MiniGPT-Forge: Build, Train & Deploy GPT Models from Scratch
=============================================================

A lightweight, hackable, production-ready GPT implementation.

Author: Prajit Datta (https://github.com/prajitdatta)
Website: https://prajitdatta.github.io/
License: Apache 2.0
"""

__version__ = "0.4.2"
__author__ = "Prajit Datta"
__url__ = "https://github.com/prajitdatta/MiniGPT-Forge"
__license__ = "Apache-2.0"

from minigpt.model.gpt import MiniGPT
from minigpt.training.trainer import Trainer
from minigpt.configs import (
    GPT2SmallConfig,
    GPT2MediumConfig,
    GPT2LargeConfig,
    GPT2XLConfig,
)

__all__ = [
    "MiniGPT",
    "Trainer",
    "GPT2SmallConfig",
    "GPT2MediumConfig",
    "GPT2LargeConfig",
    "GPT2XLConfig",
]
