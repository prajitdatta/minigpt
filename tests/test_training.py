"""
Training Tests
==============

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from minigpt.model.gpt import MiniGPT
from minigpt.training.optimizer import configure_optimizer
from minigpt.training.scheduler import get_scheduler, WarmupCosineScheduler
from minigpt.training.callbacks import EarlyStoppingCallback
from minigpt.utils.config import ModelConfig


@pytest.fixture
def tiny_model():
    config = ModelConfig(
        vocab_size=64, d_model=32, n_heads=2, n_layers=1,
        d_ff=128, block_size=16, dropout=0.0,
    )
    return MiniGPT(config)


@pytest.fixture
def tiny_dataloader():
    x = torch.randint(0, 64, (32, 16))
    y = torch.randint(0, 64, (32, 16))
    return DataLoader(TensorDataset(x, y), batch_size=8)


class TestOptimizer:
    def test_configure_optimizer(self, tiny_model):
        opt = configure_optimizer(tiny_model, learning_rate=1e-3, weight_decay=0.1)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 0.1
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_optimizer_step(self, tiny_model):
        opt = configure_optimizer(tiny_model, learning_rate=1e-3)
        x = torch.randint(0, 64, (4, 16))
        y = torch.randint(0, 64, (4, 16))
        _, loss = tiny_model(x, targets=y)
        loss.backward()
        opt.step()


class TestScheduler:
    def test_cosine_scheduler(self, tiny_model):
        opt = configure_optimizer(tiny_model, learning_rate=1e-3)
        scheduler = get_scheduler(opt, "cosine", total_steps=100, warmup_steps=10)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            scheduler.step()
        # LR should increase during warmup then decrease
        assert lrs[0] < lrs[10]

    def test_warmup_cosine(self, tiny_model):
        opt = configure_optimizer(tiny_model, learning_rate=1e-3)
        scheduler = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
        for _ in range(50):
            scheduler.step()
        assert opt.param_groups[0]["lr"] < 1e-3


class TestCallbacks:
    def test_early_stopping(self):
        cb = EarlyStoppingCallback(patience=3)

        class FakeTrainer:
            max_epochs = 100

        trainer = FakeTrainer()
        for i in range(10):
            cb.on_epoch_end(trainer, i, 1.0, 1.0)  # No improvement

        assert trainer.max_epochs < 100  # Should have triggered
