"""
Token and Positional Embeddings
================================

Implements learned token embeddings and positional encodings
with support for both absolute and RoPE positional embeddings.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learned token embedding lookup table.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding dimensionality.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Integer tensor of shape (batch_size, seq_len).

        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        return self.embedding(token_ids)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned absolute positional embedding.

    Args:
        max_seq_len: Maximum sequence length supported.
        d_model: Embedding dimensionality.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, positions: Optional[torch.Tensor] = None, seq_len: int = 0) -> torch.Tensor:
        """
        Args:
            positions: Optional explicit position indices.
            seq_len: If positions is None, generate [0, 1, ..., seq_len-1].

        Returns:
            Positional embeddings of shape (1, seq_len, d_model) or (batch, seq_len, d_model).
        """
        if positions is None:
            positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Does not have learnable parameters — positions are encoded using
    sine and cosine functions of different frequencies.

    Args:
        max_seq_len: Maximum sequence length.
        d_model: Model dimensionality.
        dropout: Dropout rate applied to embeddings.
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            x + positional encoding, same shape as input.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al., 2021.

    Encodes position by rotating query and key vectors in 2D subspaces.
    More effective for long-range dependencies than absolute PE.

    Args:
        head_dim: Dimensionality of each attention head.
        max_seq_len: Maximum sequence length.
        base: Base for the frequency computation (default: 10000).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cache", emb.sin().unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by splitting in half and swapping."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim).
            k: Key tensor (batch, n_heads, seq_len, head_dim).
            offset: Position offset for KV-cache scenarios.

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes.
        """
        seq_len = q.size(2)
        cos = self.cos_cache[:, :, offset : offset + seq_len, :]
        sin = self.sin_cache[:, :, offset : offset + seq_len, :]

        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated


class TransformerEmbedding(nn.Module):
    """
    Combined token + positional embedding layer.

    Combines token embeddings with positional embeddings and applies
    dropout. Supports weight tying with the output projection.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimensionality.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        pos_type: Positional embedding type ('learned', 'sinusoidal').
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pos_type: str = "learned",
    ):
        super().__init__()

        self.token_emb = TokenEmbedding(vocab_size, d_model)

        if pos_type == "learned":
            self.pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)
        elif pos_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(max_seq_len, d_model, dropout=0.0)
        else:
            raise ValueError(f"Unknown positional embedding type: {pos_type}")

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Integer tensor of shape (batch_size, seq_len).

        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        B, T = token_ids.size()
        tok_emb = self.token_emb(token_ids)

        if isinstance(self.pos_emb, LearnedPositionalEmbedding):
            pos_emb = self.pos_emb(seq_len=T)
            x = tok_emb + pos_emb
        else:
            x = self.pos_emb(tok_emb)

        return self.dropout(x)
