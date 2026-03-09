<p align="center">
  <img src="https://static.wixstatic.com/media/68ad1b_96c952149d584505bdfc30a3cbf36795~mv2.jpg" width="120" height="120" style="border-radius: 50%;" alt="MiniGPT-Forge Logo"/>
</p>

<h1 align="center">⚡ MiniGPT-Forge</h1>

<p align="center">
  <strong>Build, Train & Deploy GPT Models from Scratch — Lightweight, Hackable, Production-Ready</strong>
</p>

<p align="center">
  <a href="https://github.com/prajitdatta/MiniGPT-Forge/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen?style=flat-square" alt="Build Status"/></a>
  <a href="https://pypi.org/project/minigpt-forge/"><img src="https://img.shields.io/badge/pypi-v0.4.2-blue?style=flat-square" alt="PyPI Version"/></a>
  <a href="https://github.com/prajitdatta/MiniGPT-Forge/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-orange?style=flat-square" alt="License"/></a>
  <a href="https://prajitdatta.github.io/"><img src="https://img.shields.io/badge/docs-website-blueviolet?style=flat-square" alt="Documentation"/></a>
  <a href="https://github.com/prajitdatta/MiniGPT-Forge/stargazers"><img src="https://img.shields.io/badge/stars-⭐-yellow?style=flat-square" alt="Stars"/></a>
  <a href="https://github.com/prajitdatta"><img src="https://img.shields.io/badge/author-Prajit%20Datta-red?style=flat-square" alt="Author"/></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-training">Training</a> •
  <a href="#-api-server">API Server</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-documentation">Docs</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🧠 What is MiniGPT-Forge?

**MiniGPT-Forge** is a from-scratch implementation of GPT-style transformer language models designed for **education, research, and rapid prototyping**. Unlike bloated frameworks, MiniGPT-Forge gives you a clean, modular, and deeply documented codebase where every line of code is intentional and understandable.

Train a **124M parameter GPT-2 class model** on a single GPU in hours, or scale up to **1.5B+ parameters** across multi-GPU clusters — all with the same elegant API.

```python
from minigpt.model import MiniGPT
from minigpt.training import Trainer
from minigpt.configs import GPT2SmallConfig

model = MiniGPT(GPT2SmallConfig())
trainer = Trainer(model, dataset="openwebtext", batch_size=32)
trainer.train(epochs=5)

# Generate text
output = model.generate("The future of AI is", max_tokens=200, temperature=0.8)
print(output)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏗️ **From-Scratch Transformers** | Pure PyTorch implementation — no hidden abstractions |
| 🔥 **Multi-Head Self-Attention** | Efficient causal attention with Flash Attention v2 support |
| 📦 **Pre-built Configs** | GPT-2 Small (124M), Medium (355M), Large (774M), XL (1.5B) |
| 🚀 **Mixed Precision Training** | FP16/BF16 with automatic gradient scaling |
| 🌐 **Distributed Training** | DDP + FSDP support for multi-GPU / multi-node training |
| 📊 **Weights & Biases Integration** | Real-time training dashboards and experiment tracking |
| 🧮 **Custom Tokenizer** | BPE tokenizer with training pipeline, or use tiktoken |
| 🔌 **REST API Server** | FastAPI-powered inference server with streaming support |
| 📓 **Jupyter Notebooks** | Interactive walkthroughs for every major concept |
| ✅ **Comprehensive Tests** | 95%+ test coverage across model, training, and data |
| 🪝 **Callback System** | Extensible hooks for logging, checkpointing, and evaluation |
| 📈 **Learning Rate Schedulers** | Cosine annealing, linear warmup, and custom schedules |

---

## 📁 Project Structure

```
MiniGPT-Forge/
├── minigpt/                    # Core library
│   ├── __init__.py             # Package initialization & version
│   ├── model/                  # Model architecture
│   │   ├── __init__.py
│   │   ├── transformer.py      # Core transformer block
│   │   ├── attention.py        # Multi-head causal self-attention
│   │   ├── feedforward.py      # Position-wise feed-forward network
│   │   ├── embeddings.py       # Token + positional embeddings
│   │   ├── gpt.py              # Full GPT model assembly
│   │   └── generation.py       # Text generation with sampling strategies
│   ├── data/                   # Data pipeline
│   │   ├── __init__.py
│   │   ├── tokenizer.py        # BPE tokenizer implementation
│   │   ├── dataset.py          # Streaming dataset with memory mapping
│   │   └── dataloader.py       # Efficient batching & collation
│   ├── training/               # Training engine
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop
│   │   ├── optimizer.py        # AdamW with weight decay fix
│   │   ├── scheduler.py        # Learning rate schedulers
│   │   ├── callbacks.py        # Training hooks & callbacks
│   │   └── distributed.py      # Multi-GPU / multi-node utilities
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py
│   │   ├── logging.py          # Structured logging
│   │   ├── checkpoint.py       # Model serialization
│   │   ├── metrics.py          # Perplexity, loss tracking
│   │   └── config.py           # Configuration management
│   ├── api/                    # Inference API
│   │   ├── __init__.py
│   │   ├── server.py           # FastAPI inference server
│   │   └── schemas.py          # Request/response schemas
│   └── configs/                # Pre-built model configs
│       ├── __init__.py
│       ├── gpt2_small.py       # 124M params
│       ├── gpt2_medium.py      # 355M params
│       ├── gpt2_large.py       # 774M params
│       └── gpt2_xl.py          # 1.5B params
├── tests/                      # Test suite
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_training.py
│   ├── test_tokenizer.py
│   ├── test_dataloader.py
│   └── test_api.py
├── scripts/                    # Utility scripts
│   ├── train.py                # CLI training entrypoint
│   ├── generate.py             # CLI generation tool
│   ├── benchmark.py            # Performance benchmarking
│   ├── convert_weights.py      # HuggingFace ↔ MiniGPT conversion
│   └── download_data.py        # Dataset downloader
├── examples/                   # Example usage
│   ├── notebooks/
│   │   ├── 01_attention_explained.ipynb
│   │   ├── 02_build_gpt_from_scratch.ipynb
│   │   ├── 03_training_your_first_model.ipynb
│   │   └── 04_fine_tuning_guide.ipynb
│   └── configs/
│       ├── train_shakespeare.yaml
│       ├── train_openwebtext.yaml
│       └── train_custom_data.yaml
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md
│   ├── TRAINING_GUIDE.md
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT.md
│   └── CHANGELOG.md
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Continuous integration
│   │   └── release.yml         # Automated releases
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── setup.py                    # Package setup
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Dependencies
├── requirements-dev.txt        # Development dependencies
├── Dockerfile                  # Container support
├── Makefile                    # Common commands
├── LICENSE                     # Apache 2.0
├── CONTRIBUTING.md             # Contribution guidelines
└── CODE_OF_CONDUCT.md          # Community standards
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/prajitdatta/MiniGPT-Forge.git
cd MiniGPT-Forge

# Install with pip
pip install -e ".[dev]"

# Or install from PyPI
pip install minigpt-forge
```

### Train Your First Model

```bash
# Train on Shakespeare (quick demo — ~5 min on a single GPU)
python scripts/train.py --config examples/configs/train_shakespeare.yaml

# Train GPT-2 Small on OpenWebText
python scripts/train.py \
  --config examples/configs/train_openwebtext.yaml \
  --model gpt2-small \
  --batch-size 32 \
  --epochs 10 \
  --lr 3e-4 \
  --wandb-project minigpt-forge
```

### Generate Text

```bash
python scripts/generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "In a world where machines can think" \
  --max-tokens 256 \
  --temperature 0.7 \
  --top-k 50
```

### Python API

```python
from minigpt import MiniGPT, Trainer, GPT2SmallConfig
from minigpt.data import TextDataset, DataLoader

# Initialize model
config = GPT2SmallConfig()
model = MiniGPT(config)
print(f"Parameters: {model.num_parameters():,}")  # 124,439,808

# Load dataset
dataset = TextDataset("data/shakespeare.txt", block_size=config.block_size)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train
trainer = Trainer(
    model=model,
    train_loader=loader,
    learning_rate=3e-4,
    weight_decay=0.1,
    max_epochs=20,
    grad_clip=1.0,
    mixed_precision=True,
)
trainer.train()

# Generate
model.eval()
print(model.generate("To be or not to be", max_tokens=100, temperature=0.8))
```

---

## 🏛️ Architecture

MiniGPT-Forge implements the **decoder-only transformer** architecture (Radford et al., 2019) with modern improvements:

```
Input Tokens
     │
     ▼
┌─────────────────────┐
│  Token Embedding     │  (vocab_size × d_model)
│  + Positional Embed  │  (max_seq_len × d_model)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Transformer Block   │ × N_layers
│  ┌─────────────────┐ │
│  │ Layer Norm       │ │
│  │ Multi-Head Attn  │ │  (causal mask, d_model → d_model)
│  │ + Residual       │ │
│  │ Layer Norm       │ │
│  │ Feed-Forward     │ │  (d_model → 4×d_model → d_model)
│  │ + Residual       │ │
│  └─────────────────┘ │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Layer Norm          │
│  Linear Head         │  (d_model → vocab_size)
└─────────┬───────────┘
          │
          ▼
     Output Logits
```

### Model Configurations

| Model | Layers | Heads | d_model | FFN dim | Params | VRAM (FP16) |
|-------|--------|-------|---------|---------|--------|-------------|
| `GPT2SmallConfig` | 12 | 12 | 768 | 3072 | 124M | ~2 GB |
| `GPT2MediumConfig` | 24 | 16 | 1024 | 4096 | 355M | ~5 GB |
| `GPT2LargeConfig` | 36 | 20 | 1280 | 5120 | 774M | ~10 GB |
| `GPT2XLConfig` | 48 | 25 | 1600 | 6400 | 1.5B | ~20 GB |

---

## 🏋️ Training

### Single GPU

```bash
python scripts/train.py --config examples/configs/train_openwebtext.yaml
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --config examples/configs/train_openwebtext.yaml \
  --distributed ddp
```

### Multi-Node

```bash
torchrun --nnodes=2 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=master:29500 \
  scripts/train.py --config examples/configs/train_openwebtext.yaml \
  --distributed fsdp
```

### Training Configuration

```yaml
# examples/configs/train_openwebtext.yaml
model:
  name: gpt2-small
  vocab_size: 50257
  block_size: 1024
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.1

training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  weight_decay: 0.1
  max_epochs: 50
  warmup_steps: 2000
  grad_clip: 1.0
  mixed_precision: true
  compile: true  # torch.compile for 2x speedup

optimizer:
  name: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8

scheduler:
  name: cosine
  min_lr: 3.0e-5

data:
  dataset: openwebtext
  num_workers: 4
  pin_memory: true

logging:
  wandb_project: minigpt-forge
  log_interval: 10
  eval_interval: 500
  save_interval: 1000
```

---

## 🔌 API Server

Launch a production-ready inference API:

```bash
python -m minigpt.api.server --checkpoint checkpoints/best_model.pt --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/completions` | Generate text completion |
| `POST` | `/v1/completions/stream` | Streaming text generation (SSE) |
| `GET`  | `/v1/models` | List available models |
| `GET`  | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
  }'
```

### Response

```json
{
  "id": "gen-a1b2c3d4",
  "model": "minigpt-forge-124M",
  "choices": [
    {
      "text": "The meaning of life is a question that has puzzled philosophers...",
      "finish_reason": "length",
      "tokens_generated": 128
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 128,
    "total_tokens": 134
  }
}
```

---

## 📊 Benchmarks

Benchmarked on a single NVIDIA A100 80GB GPU:

| Metric | GPT-2 Small (124M) | GPT-2 Medium (355M) |
|--------|--------------------|-----------------------|
| Training throughput | ~48,000 tok/s | ~18,000 tok/s |
| Inference latency (128 tok) | 42ms | 89ms |
| Val perplexity (OpenWebText) | 19.8 | 15.2 |
| Memory usage (FP16) | 2.1 GB | 5.3 GB |
| Time to convergence | ~8 hours | ~28 hours |

---

## 🐳 Docker

```bash
# Build
docker build -t minigpt-forge .

# Train
docker run --gpus all -v $(pwd)/data:/app/data minigpt-forge \
  python scripts/train.py --config examples/configs/train_shakespeare.yaml

# Serve API
docker run --gpus all -p 8000:8000 minigpt-forge \
  python -m minigpt.api.server --checkpoint /app/checkpoints/model.pt
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture Deep Dive](docs/ARCHITECTURE.md) | Detailed explanation of every component |
| [Training Guide](docs/TRAINING_GUIDE.md) | End-to-end training walkthrough |
| [API Reference](docs/API_REFERENCE.md) | Complete API documentation |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment strategies |
| [Changelog](docs/CHANGELOG.md) | Version history and release notes |
| [Contributing](CONTRIBUTING.md) | How to contribute to MiniGPT-Forge |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Setup development environment
git clone https://github.com/prajitdatta/MiniGPT-Forge.git
cd MiniGPT-Forge
pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

---

## 📜 Citation

```bibtex
@software{minigpt_forge,
  author       = {Prajit Datta},
  title        = {MiniGPT-Forge: Build, Train \& Deploy GPT Models from Scratch},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/prajitdatta/MiniGPT-Forge}
}
```

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <img src="https://static.wixstatic.com/media/68ad1b_96c952149d584505bdfc30a3cbf36795~mv2.jpg" width="40" height="40" style="border-radius: 50%;" alt="Prajit Datta"/>
</p>

<p align="center">
  Built with 🔥 by <a href="https://github.com/prajitdatta"><strong>Prajit Datta</strong></a><br/>
  <a href="https://prajitdatta.github.io/">Website</a> •
  <a href="https://github.com/prajitdatta">GitHub</a> •
  <a href="https://github.com/prajitdatta/MiniGPT-Forge/issues">Report Bug</a> •
  <a href="https://github.com/prajitdatta/MiniGPT-Forge/issues">Request Feature</a>
</p>

<p align="center">
  <sub>If you find MiniGPT-Forge useful, please consider giving it a ⭐ on <a href="https://github.com/prajitdatta/MiniGPT-Forge">GitHub</a>!</sub>
</p>
