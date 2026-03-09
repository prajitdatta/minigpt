#!/usr/bin/env python3
"""
MiniGPT-Forge Training Script
===============================

CLI entrypoint for training MiniGPT models.

Usage:
    python scripts/train.py --config examples/configs/train_shakespeare.yaml
    python scripts/train.py --model gpt2-small --data data/train.bin

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import argparse
import yaml
import torch

from minigpt.model.gpt import MiniGPT
from minigpt.training.trainer import Trainer
from minigpt.data.dataset import TextDataset, MemmapDataset
from minigpt.data.dataloader import create_dataloader
from minigpt.data.tokenizer import TiktokenWrapper
from minigpt.utils.config import get_config, ModelConfig, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MiniGPT model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, default="gpt2-small", help="Model config name")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--distributed", type=str, choices=["ddp", "fsdp"], default=None)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Load config from YAML or CLI
    if args.config:
        cfg = load_yaml_config(args.config)
        model_cfg = ModelConfig(**cfg.get("model", {}))
        train_cfg = cfg.get("training", {})
    else:
        model_cfg = get_config(args.model)
        train_cfg = {}

    # Override with CLI args
    batch_size = train_cfg.get("batch_size", args.batch_size)
    max_epochs = train_cfg.get("max_epochs", args.epochs)
    learning_rate = train_cfg.get("learning_rate", args.lr)

    print(f"Model: {model_cfg.name}")
    print(f"Config: d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}, n_heads={model_cfg.n_heads}")

    # Initialize model
    model = MiniGPT(model_cfg)

    # Setup tokenizer
    tokenizer = TiktokenWrapper("gpt2")

    # Load data
    data_path = args.data or train_cfg.get("data", {}).get("path", "data/train.bin")
    if data_path.endswith(".bin"):
        train_dataset = MemmapDataset(data_path, block_size=model_cfg.block_size)
    else:
        train_dataset = TextDataset(data_path, block_size=model_cfg.block_size, tokenizer=tokenizer)

    val_dataset = None
    val_path = args.val_data or data_path.replace("train", "val")
    try:
        if val_path.endswith(".bin"):
            val_dataset = MemmapDataset(val_path, block_size=model_cfg.block_size)
        else:
            val_dataset = TextDataset(val_path, block_size=model_cfg.block_size, tokenizer=tokenizer)
    except FileNotFoundError:
        print("No validation data found, skipping validation.")

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if val_dataset else None

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        wandb_project=args.wandb_project,
        compile_model=args.compile,
    )

    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
