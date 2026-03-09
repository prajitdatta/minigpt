#!/usr/bin/env python3
"""
MiniGPT-Forge Text Generation Script
======================================

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import argparse
import torch
from minigpt.model.gpt import MiniGPT
from minigpt.data.tokenizer import TiktokenWrapper


def main():
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.checkpoint}...")
    model = MiniGPT.from_pretrained(args.checkpoint, device=device)
    tokenizer = TiktokenWrapper("gpt2")

    print(f"Generating with prompt: '{args.prompt}'")
    print("-" * 60)

    output = model.generate(
        prompt=args.prompt,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        stream=args.stream,
    )

    if not args.stream:
        print(output)
    print()


if __name__ == "__main__":
    main()
