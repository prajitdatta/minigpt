"""
Training Callbacks
==================

Extensible callback system for hooking into the training loop.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

from typing import List, Optional


class Callback:
    """Base callback class. Override methods to hook into training events."""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch: int):
        pass

    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: Optional[float]):
        pass

    def on_step_begin(self, trainer, step: int):
        pass

    def on_step_end(self, trainer, step: int, loss: float):
        pass


class CallbackList:
    """Manages a list of callbacks and dispatches events."""

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer, epoch: int):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer, epoch: int, train_loss: float, val_loss: Optional[float]):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, train_loss, val_loss)

    def on_step_begin(self, trainer, step: int):
        for cb in self.callbacks:
            cb.on_step_begin(trainer, step)

    def on_step_end(self, trainer, step: int, loss: float):
        for cb in self.callbacks:
            cb.on_step_end(trainer, step, loss)


class EarlyStoppingCallback(Callback):
    """
    Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        if val_loss is None:
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                trainer.max_epochs = epoch + 1  # Stop training


class PrintCallback(Callback):
    """Simple callback that prints training progress."""

    def on_train_begin(self, trainer):
        print("=" * 60)
        print("Training started")
        print("=" * 60)

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        msg = f"Epoch {epoch + 1:>3d} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        print(msg)

    def on_train_end(self, trainer):
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)
