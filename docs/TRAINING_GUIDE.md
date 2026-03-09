# Training Guide

<p align="center">
  <img src="https://static.wixstatic.com/media/68ad1b_96c952149d584505bdfc30a3cbf36795~mv2.jpg" width="60" alt="MiniGPT-Forge"/>
</p>

## Quick Start: Shakespeare in 5 Minutes

```bash
# Download data
python scripts/download_data.py shakespeare

# Train
python scripts/train.py --config examples/configs/train_shakespeare.yaml
```

## Data Preparation

MiniGPT-Forge supports three data formats:

### Plain Text Files
Simplest option — just provide a `.txt` file. The tokenizer handles everything.

### Memory-Mapped Binary
For large datasets, pre-tokenize into `.bin` files:

```python
from minigpt.data import prepare_data, TiktokenWrapper

tokenizer = TiktokenWrapper("gpt2")
prepare_data("my_corpus.txt", "data/processed/", tokenizer, val_split=0.1)
```

### Sharded Datasets
For very large datasets, split into shards for streaming:

```python
from minigpt.data import StreamingDataset
dataset = StreamingDataset("data/shards/", block_size=1024)
```

## Training Strategies

### Mixed Precision
Enabled by default. Uses BF16 on Ampere+ GPUs, FP16 otherwise.

### Gradient Accumulation
Simulate larger batch sizes:
```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 8  # Effective batch size = 128
```

### Multi-GPU Training
```bash
# DDP (Data Distributed Parallel)
torchrun --nproc_per_node=4 scripts/train.py --distributed ddp

# FSDP (Fully Sharded Data Parallel) for large models
torchrun --nproc_per_node=8 scripts/train.py --distributed fsdp
```

### torch.compile
Enable for ~2x speedup on PyTorch 2.0+:
```yaml
training:
  compile: true
```

## Monitoring with Weights & Biases

```bash
python scripts/train.py --wandb-project minigpt-forge
```

## Hyperparameter Guide

| Model Size | Learning Rate | Batch Size | Weight Decay | Warmup Steps |
|-----------|--------------|-----------|-------------|-------------|
| Small (124M) | 3e-4 | 32-64 | 0.1 | 2000 |
| Medium (355M) | 2e-4 | 16-32 | 0.1 | 3000 |
| Large (774M) | 1.5e-4 | 8-16 | 0.1 | 4000 |
| XL (1.5B) | 1e-4 | 4-8 | 0.1 | 5000 |

---

<p align="center">
  <a href="https://github.com/prajitdatta/MiniGPT-Forge">Back to Repository</a> •
  Built by <a href="https://github.com/prajitdatta">Prajit Datta</a>
</p>
