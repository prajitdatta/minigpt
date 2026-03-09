"""
Attention Tests
===============

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import pytest
import torch
from minigpt.model.attention import CausalSelfAttention, MultiQueryAttention


class TestMultiQueryAttention:
    def test_output_shape(self):
        mqa = MultiQueryAttention(d_model=128, n_heads=4, block_size=64, dropout=0.0)
        x = torch.randn(2, 16, 128)
        out = mqa(x)
        assert out.shape == (2, 16, 128)

    def test_fewer_kv_params(self):
        """MQA should have fewer parameters than standard MHA."""
        mha = CausalSelfAttention(d_model=128, n_heads=4, block_size=64)
        mqa = MultiQueryAttention(d_model=128, n_heads=4, block_size=64)
        mha_params = sum(p.numel() for p in mha.parameters())
        mqa_params = sum(p.numel() for p in mqa.parameters())
        assert mqa_params < mha_params
