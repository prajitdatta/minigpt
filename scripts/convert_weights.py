#!/usr/bin/env python3
"""
Weight Conversion: HuggingFace ↔ MiniGPT-Forge
=================================================

Convert between HuggingFace GPT-2 weights and MiniGPT-Forge format.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import argparse
import torch
from minigpt.model.gpt import MiniGPT
from minigpt.utils.config import get_config


def from_huggingface(hf_model_name: str = "gpt2", output_path: str = "checkpoints/gpt2_converted.pt"):
    """Convert HuggingFace GPT-2 weights to MiniGPT-Forge format."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError("transformers is required: pip install transformers")

    print(f"Loading HuggingFace model: {hf_model_name}")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_sd = hf_model.state_dict()

    # Determine config
    config_map = {"gpt2": "gpt2-small", "gpt2-medium": "gpt2-medium",
                  "gpt2-large": "gpt2-large", "gpt2-xl": "gpt2-xl"}
    config = get_config(config_map.get(hf_model_name, "gpt2-small"))
    model = MiniGPT(config)

    # Map keys
    key_mapping = {
        "transformer.wte.weight": "embeddings.token_emb.embedding.weight",
        "transformer.wpe.weight": "embeddings.pos_emb.embedding.weight",
        "transformer.ln_f.weight": "ln_f.weight",
        "transformer.ln_f.bias": "ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }

    for i in range(config.n_layers):
        pfx_hf = f"transformer.h.{i}"
        pfx_mg = f"blocks.{i}"
        key_mapping.update({
            f"{pfx_hf}.ln_1.weight": f"{pfx_mg}.ln1.weight",
            f"{pfx_hf}.ln_1.bias": f"{pfx_mg}.ln1.bias",
            f"{pfx_hf}.ln_2.weight": f"{pfx_mg}.ln2.weight",
            f"{pfx_hf}.ln_2.bias": f"{pfx_mg}.ln2.bias",
            f"{pfx_hf}.attn.c_attn.weight": f"{pfx_mg}.attn.qkv_proj.weight",
            f"{pfx_hf}.attn.c_attn.bias": f"{pfx_mg}.attn.qkv_proj.bias",
            f"{pfx_hf}.attn.c_proj.weight": f"{pfx_mg}.attn.out_proj.weight",
            f"{pfx_hf}.attn.c_proj.bias": f"{pfx_mg}.attn.out_proj.bias",
            f"{pfx_hf}.mlp.c_fc.weight": f"{pfx_mg}.ffn.w1.weight",
            f"{pfx_hf}.mlp.c_fc.bias": f"{pfx_mg}.ffn.w1.bias",
            f"{pfx_hf}.mlp.c_proj.weight": f"{pfx_mg}.ffn.w2.weight",
            f"{pfx_hf}.mlp.c_proj.bias": f"{pfx_mg}.ffn.w2.bias",
        })

    new_sd = {}
    for hf_key, mg_key in key_mapping.items():
        if hf_key in hf_sd:
            w = hf_sd[hf_key]
            # HF GPT-2 uses Conv1D (transposed weights)
            if "attn" in hf_key and "weight" in hf_key:
                w = w.t()
            if "mlp" in hf_key and "weight" in hf_key:
                w = w.t()
            new_sd[mg_key] = w

    model.load_state_dict(new_sd, strict=False)
    model.save_checkpoint(output_path)
    print(f"Converted model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert model weights")
    parser.add_argument("--from-hf", type=str, help="HuggingFace model name (e.g., 'gpt2')")
    parser.add_argument("--output", type=str, default="checkpoints/converted.pt")
    args = parser.parse_args()

    if args.from_hf:
        from_huggingface(args.from_hf, args.output)


if __name__ == "__main__":
    main()
