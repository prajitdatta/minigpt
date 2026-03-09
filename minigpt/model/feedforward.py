"""
Position-wise Feed-Forward Network
===================================

Implements the FFN sub-layer of the transformer block with
support for GELU activation and optional gated variants (SwiGLU).

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applies two linear transformations with a GELU activation in between:
        FFN(x) = W2 * GELU(W1 * x + b1) + b2

    Args:
        d_model: Input and output dimensionality.
        d_ff: Hidden layer dimensionality (typically 4 * d_model).
        dropout: Dropout probability.
        bias: Whether to use bias terms.
        activation: Activation function ('gelu', 'relu', 'swiglu').
    """

    def __init__(
        self,
        d_model: int = 768,
        d_ff: int = 3072,
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        if activation == "swiglu":
            # SwiGLU: gate and value projections
            self.w1 = nn.Linear(d_model, d_ff, bias=bias)
            self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
            self.w2 = nn.Linear(d_ff, d_model, bias=bias)
            self._activation = self._swiglu
        else:
            self.w1 = nn.Linear(d_model, d_ff, bias=bias)
            self.w2 = nn.Linear(d_ff, d_model, bias=bias)
            self.w_gate = None

            if activation == "gelu":
                self._activation = lambda x: F.gelu(x, approximate="tanh")
            elif activation == "relu":
                self._activation = F.relu
            else:
                raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: SiLU(Wx) * Vx"""
        return F.silu(self.w1(x)) * self.w_gate(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        if self.w_gate is not None:
            # SwiGLU path
            hidden = self._activation(x)
        else:
            hidden = self._activation(self.w1(x))

        return self.dropout(self.w2(hidden))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
