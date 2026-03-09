"""
Multi-Head Causal Self-Attention
================================

Implements scaled dot-product attention with causal masking,
optional Flash Attention v2 support, and KV-cache for inference.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.

    Uses a causal mask to prevent attending to future tokens,
    making it suitable for autoregressive language modeling.

    Args:
        d_model: Dimensionality of the model (embedding size).
        n_heads: Number of attention heads.
        block_size: Maximum sequence length.
        dropout: Dropout probability for attention weights.
        bias: Whether to use bias in linear projections.
        flash: Whether to use Flash Attention (requires PyTorch 2.0+).
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        block_size: int = 1024,
        dropout: float = 0.1,
        bias: bool = True,
        flash: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.flash = flash and hasattr(F, "scaled_dot_product_attention")

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask — registered as buffer (not a parameter)
        if not self.flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

        # KV cache for efficient autoregressive inference
        self._kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            use_cache: If True, use and update KV cache for inference.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.size()

        # Compute Q, K, V in a single matmul
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # KV cache for autoregressive generation
        if use_cache:
            if self._kv_cache is not None:
                prev_k, prev_v = self._kv_cache
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            self._kv_cache = (k.detach(), v.detach())

        # Attention computation
        if self.flash:
            # Use PyTorch 2.0 Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True if not use_cache else False,
            )
        else:
            # Manual attention with causal mask
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply causal mask
            if not use_cache:
                attn_weights = attn_weights.masked_fill(
                    self.causal_mask[:, :, :T, :T] == 0, float("-inf")
                )

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (B, n_heads, T, head_dim) → (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        output = self.resid_dropout(self.out_proj(attn_output))

        return output

    def reset_cache(self):
        """Clear the KV cache."""
        self._kv_cache = None

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, dropout={self.dropout}, "
            f"flash={self.flash}"
        )


class MultiQueryAttention(CausalSelfAttention):
    """
    Multi-Query Attention (MQA) variant.

    Uses a single key-value head shared across all query heads,
    reducing KV cache memory and improving inference throughput.
    Ref: Shazeer (2019) "Fast Transformer Decoding"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        d_model = kwargs.get("d_model", 768)
        n_heads = kwargs.get("n_heads", 12)
        bias = kwargs.get("bias", True)

        self.head_dim = d_model // n_heads

        # Override: separate Q (multi-head) and KV (single-head)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * self.head_dim, bias=bias)

        # Remove the combined qkv_proj from parent
        del self.qkv_proj

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)

        # Single head for K, V — expand to match query heads
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2).expand(-1, self.n_heads, -1, -1)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2).expand(-1, self.n_heads, -1, -1)

        if use_cache and self._kv_cache is not None:
            prev_k, prev_v = self._kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        if use_cache:
            self._kv_cache = (k.detach(), v.detach())

        if self.flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True if not use_cache else False,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            S = k.size(2)
            if not use_cache:
                mask = torch.tril(torch.ones(T, S, device=x.device)).view(1, 1, T, S)
                attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(attn_output))
