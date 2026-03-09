"""
Transformer Block
=================

Implements a single transformer decoder block with
pre-norm architecture (LayerNorm before attention and FFN).

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import torch
import torch.nn as nn

from minigpt.model.attention import CausalSelfAttention
from minigpt.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Uses pre-norm (LayerNorm before sublayers) architecture,
    which is more stable for training deep networks.

    Architecture:
        x → LayerNorm → Attention → + residual
          → LayerNorm → FFN       → + residual

    Args:
        d_model: Model dimensionality.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimensionality.
        block_size: Maximum sequence length.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        activation: FFN activation function.
        flash: Whether to use Flash Attention.
        layer_idx: Index of this block (for logging/debugging).
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        block_size: int = 1024,
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = "gelu",
        flash: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        # Pre-norm layers
        self.ln1 = nn.LayerNorm(d_model, bias=bias)
        self.ln2 = nn.LayerNorm(d_model, bias=bias)

        # Self-attention
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
            flash=flash,
        )

        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            use_cache: Whether to use KV cache for inference.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), use_cache=use_cache)

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x

    def reset_cache(self):
        """Clear the attention KV cache."""
        self.attn.reset_cache()

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}"
