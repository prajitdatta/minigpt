"""
Training Engine
===============

Main training loop with mixed precision, gradient accumulation,
learning rate scheduling, checkpointing, and W&B integration.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import os
import time
import math
from typing import Optional, List, Callable
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from minigpt.training.optimizer import configure_optimizer
from minigpt.training.scheduler import get_scheduler
from minigpt.training.callbacks import Callback, CallbackList
from minigpt.utils.logging import get_logger
from minigpt.utils.metrics import MetricsTracker

logger = get_logger(__name__)


class Trainer:
    """
    Complete training engine for MiniGPT models.

    Handles the full training pipeline including:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Checkpointing and resume
    - Weights & Biases logging
    - Validation evaluation
    - Callback hooks

    Args:
        model: The MiniGPT model to train.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay for AdamW.
        max_epochs: Maximum number of training epochs.
        grad_clip: Maximum gradient norm for clipping.
        mixed_precision: Whether to use mixed precision.
        gradient_accumulation_steps: Number of accumulation steps.
        warmup_steps: Number of warmup steps for LR scheduler.
        min_lr: Minimum learning rate (for cosine scheduler).
        scheduler_type: Type of LR scheduler ('cosine', 'linear', 'constant').
        log_interval: Steps between logging.
        eval_interval: Steps between evaluation.
        save_interval: Steps between checkpoints.
        checkpoint_dir: Directory for saving checkpoints.
        wandb_project: W&B project name (None to disable).
        wandb_run_name: W&B run name.
        callbacks: List of callback instances.
        compile_model: Whether to use torch.compile.
        device: Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_epochs: int = 50,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 2000,
        min_lr: float = 3e-5,
        scheduler_type: str = "cosine",
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        checkpoint_dir: str = "checkpoints",
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        compile_model: bool = False,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Optional torch.compile
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Optimizer
        self.optimizer = configure_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * max_epochs // gradient_accumulation_steps
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )

        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler(enabled=mixed_precision)
        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # Metrics
        self.metrics = MetricsTracker()

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # W&B
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "max_epochs": max_epochs,
                        "batch_size": train_loader.batch_size,
                        "model_params": sum(p.numel() for p in model.parameters()),
                    },
                )
            except ImportError:
                logger.warning("wandb not installed. Skipping W&B logging.")

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")

        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Total training steps: {total_steps:,}")

    def train(self):
        """Execute the full training loop."""
        logger.info("Starting training...")
        self.callbacks.on_train_begin(self)

        for epoch in range(self.max_epochs):
            self.callbacks.on_epoch_begin(self, epoch)
            epoch_loss = self._train_epoch(epoch)

            # Validation
            val_loss = None
            if self.val_loader is not None and (epoch + 1) % 1 == 0:
                val_loss = self._evaluate()
                logger.info(f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt", epoch, val_loss)
            else:
                logger.info(f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f}")

            self.callbacks.on_epoch_end(self, epoch, epoch_loss, val_loss)

        self.callbacks.on_train_end(self)

        if self.wandb_run:
            self.wandb_run.finish()

        logger.info("Training complete!")

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Mixed precision forward pass
            ctx = (
                autocast(device_type=self.device.type, dtype=self.amp_dtype)
                if self.mixed_precision
                else nullcontext()
            )

            with ctx:
                logits, loss = self.model(x, targets=y)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            batch_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += batch_loss
            num_batches += 1

            # Logging
            if self.global_step % self.log_interval == 0 and self.global_step > 0:
                elapsed = time.time() - t0
                tokens_per_sec = (
                    self.log_interval
                    * x.size(0)
                    * x.size(1)
                    * self.gradient_accumulation_steps
                    / elapsed
                )
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {self.global_step:>6d} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:,.0f}"
                )

                if self.wandb_run:
                    self.wandb_run.log({
                        "train/loss": batch_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": self.global_step,
                    })

                t0 = time.time()

            # Evaluation
            if (
                self.eval_interval > 0
                and self.global_step % self.eval_interval == 0
                and self.global_step > 0
                and self.val_loader is not None
            ):
                val_loss = self._evaluate()
                logger.info(f"Step {self.global_step} | Val Loss: {val_loss:.4f}")

                if self.wandb_run:
                    self.wandb_run.log({
                        "val/loss": val_loss,
                        "val/perplexity": math.exp(val_loss),
                        "val/step": self.global_step,
                    })

                self.model.train()

            # Checkpointing
            if (
                self.save_interval > 0
                and self.global_step % self.save_interval == 0
                and self.global_step > 0
            ):
                self._save_checkpoint(
                    f"step_{self.global_step}.pt", epoch, total_loss / num_batches
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            ctx = (
                autocast(device_type=self.device.type, dtype=self.amp_dtype)
                if self.mixed_precision
                else nullcontext()
            )

            with ctx:
                _, loss = self.model(x, targets=y)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save a training checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        self.model.save_checkpoint(
            path,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            loss=loss,
        )
        logger.info(f"Checkpoint saved: {path}")

    @classmethod
    def from_config(cls, model, train_loader, config, **kwargs):
        """Create a Trainer from a TrainingConfig object."""
        return cls(
            model=model,
            train_loader=train_loader,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_epochs=config.max_epochs,
            grad_clip=config.grad_clip,
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
            scheduler_type=config.scheduler,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            save_interval=config.save_interval,
            checkpoint_dir=config.checkpoint_dir,
            wandb_project=config.wandb_project,
            wandb_run_name=config.wandb_run_name,
            compile_model=config.compile_model,
            **kwargs,
        )
