# Changelog

All notable changes to MiniGPT-Forge are documented here.

## [0.4.2] - 2026-03-09

### Added
- Multi-Query Attention (MQA) variant for faster inference
- SwiGLU activation option for feed-forward network
- Rotary Positional Embeddings (RoPE) support
- Streaming SSE endpoint for real-time generation
- Comprehensive test suite (95%+ coverage)

### Improved
- Flash Attention v2 integration with PyTorch 2.0+
- KV-cache efficiency for autoregressive generation
- Training throughput with `torch.compile` support
- Documentation with interactive notebooks

## [0.3.0] - 2026-01-15

### Added
- FastAPI inference server
- FSDP distributed training support
- Weights & Biases integration
- HuggingFace weight conversion script

### Fixed
- Memory leak in long-context generation
- Gradient accumulation step counting

## [0.2.0] - 2025-10-01

### Added
- BPE tokenizer with training pipeline
- Memory-mapped datasets for large corpora
- Mixed precision training (FP16/BF16)
- Cosine annealing with warmup scheduler

## [0.1.0] - 2025-07-15

### Initial Release
- Core GPT model implementation
- Basic training loop
- Character-level dataset support
- GPT-2 Small/Medium/Large/XL configs

---

Maintained by [Prajit Datta](https://github.com/prajitdatta) • [Website](https://prajitdatta.github.io/)
