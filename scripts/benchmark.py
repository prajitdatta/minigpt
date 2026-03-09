#!/usr/bin/env python3
"""
MiniGPT-Forge Benchmarking Script
===================================

Measures training throughput and inference latency.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import time
import torch
from minigpt.model.gpt import MiniGPT
from minigpt.utils.config import get_config


def benchmark_forward(model, device, batch_size=16, seq_len=512, n_iters=50):
    """Benchmark forward pass throughput."""
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    tokens_per_sec = (batch_size * seq_len * n_iters) / elapsed
    ms_per_batch = (elapsed / n_iters) * 1000

    return tokens_per_sec, ms_per_batch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in ["gpt2-small", "gpt2-medium"]:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'=' * 60}")

        config = get_config(model_name)
        model = MiniGPT(config).to(device).eval()

        tok_s, ms = benchmark_forward(model, device)
        print(f"  Forward throughput: {tok_s:,.0f} tokens/sec")
        print(f"  Latency per batch:  {ms:.1f} ms")

        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak GPU memory:    {mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
