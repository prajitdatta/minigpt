# Architecture Deep Dive

<p align="center">
  <img src="https://static.wixstatic.com/media/68ad1b_96c952149d584505bdfc30a3cbf36795~mv2.jpg" width="60" alt="MiniGPT-Forge"/>
</p>

## Overview

MiniGPT-Forge implements a **decoder-only transformer** architecture following the GPT-2 design with modern improvements. This document explains every component in detail.

## Component Stack

### 1. Token Embeddings (`minigpt/model/embeddings.py`)

Converts integer token IDs into dense vectors. Uses a standard `nn.Embedding` lookup table of shape `(vocab_size, d_model)`.

### 2. Positional Embeddings

Supports three strategies:

- **Learned** (default): Trainable embedding table of shape `(max_seq_len, d_model)`, same as GPT-2.
- **Sinusoidal**: Fixed sine/cosine encoding from the original Transformer paper.
- **Rotary (RoPE)**: Rotation-based encoding that encodes relative positions by rotating query and key vectors.

### 3. Multi-Head Causal Self-Attention (`minigpt/model/attention.py`)

The core of the transformer. Implements:

- Combined QKV projection for efficiency
- Causal masking to prevent attending to future tokens
- Flash Attention v2 support via `F.scaled_dot_product_attention`
- KV-cache for efficient autoregressive generation
- Multi-Query Attention (MQA) variant for faster inference

### 4. Feed-Forward Network (`minigpt/model/feedforward.py`)

Position-wise FFN with support for:

- **GELU** activation (default, GPT-2 style)
- **SwiGLU** activation (Llama-style gated FFN)
- **ReLU** activation

### 5. Transformer Block (`minigpt/model/transformer.py`)

Pre-norm architecture:
```
x → LayerNorm → Attention → + residual → LayerNorm → FFN → + residual
```

### 6. Language Model Head

Linear projection from `d_model` to `vocab_size` with optional weight tying to the token embedding matrix.

## Weight Initialization

Follows GPT-2 initialization:
- All linear layers: `N(0, 0.02)`
- Residual projections: scaled by `1/√(2 * n_layers)`
- LayerNorm: weight=1, bias=0

## Generation Strategies

See `minigpt/model/generation.py`:
- Greedy decoding
- Temperature sampling
- Top-k filtering
- Nucleus (top-p) sampling
- Repetition penalty
- Frequency/presence penalties

---

<p align="center">
  <a href="https://github.com/prajitdatta/MiniGPT-Forge">Back to Repository</a> •
  Built by <a href="https://github.com/prajitdatta">Prajit Datta</a>
</p>
