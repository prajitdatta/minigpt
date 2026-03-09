"""
Model Tests
============

Comprehensive tests for MiniGPT model components.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import pytest
import torch

from minigpt.model.gpt import MiniGPT
from minigpt.model.attention import CausalSelfAttention, MultiQueryAttention
from minigpt.model.feedforward import FeedForward
from minigpt.model.transformer import TransformerBlock
from minigpt.model.embeddings import TransformerEmbedding, RotaryPositionalEmbedding
from minigpt.utils.config import ModelConfig, GPT2SmallConfig


@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        block_size=64,
        dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(d_model=128, n_heads=4, block_size=64, dropout=0.0)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_causal_masking(self):
        attn = CausalSelfAttention(
            d_model=128, n_heads=4, block_size=64, dropout=0.0, flash=False
        )
        x = torch.randn(1, 8, 128)
        out = attn(x)
        assert out.shape == (1, 8, 128)

    def test_kv_cache(self):
        attn = CausalSelfAttention(d_model=128, n_heads=4, block_size=64, dropout=0.0)
        x1 = torch.randn(1, 4, 128)
        x2 = torch.randn(1, 1, 128)
        attn.reset_cache()
        _ = attn(x1, use_cache=True)
        out = attn(x2, use_cache=True)
        assert out.shape == (1, 1, 128)


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(d_model=128, d_ff=512)
        x = torch.randn(2, 16, 128)
        assert ffn(x).shape == (2, 16, 128)

    def test_swiglu(self):
        ffn = FeedForward(d_model=128, d_ff=512, activation="swiglu")
        x = torch.randn(2, 16, 128)
        assert ffn(x).shape == (2, 16, 128)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, block_size=64)
        x = torch.randn(2, 16, 128)
        assert block(x).shape == (2, 16, 128)

    def test_residual_connection(self):
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512, block_size=64, dropout=0.0)
        x = torch.zeros(1, 4, 128)
        out = block(x)
        # Output should not be zero due to bias terms
        assert not torch.allclose(out, x)


class TestMiniGPT:
    def test_forward(self, small_config):
        model = MiniGPT(small_config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_forward_with_targets(self, small_config):
        model = MiniGPT(small_config)
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        logits, loss = model(x, targets=y)
        assert logits.shape == (2, 16, 256)
        assert loss.item() > 0

    def test_parameter_count(self, small_config):
        model = MiniGPT(small_config)
        n_params = model.num_parameters()
        assert n_params > 0

    def test_weight_tying(self, small_config):
        small_config.weight_tying = True
        model = MiniGPT(small_config)
        assert model.lm_head.weight is model.embeddings.token_emb.embedding.weight

    def test_gpt2_small_config(self):
        config = GPT2SmallConfig()
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12

    def test_save_load(self, small_config, tmp_path):
        model = MiniGPT(small_config)
        path = str(tmp_path / "test_model.pt")
        model.save_checkpoint(path)
        loaded = MiniGPT.from_pretrained(path)
        assert loaded.num_parameters() == model.num_parameters()


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryPositionalEmbedding(head_dim=32, max_seq_len=64)
        q = torch.randn(1, 4, 16, 32)
        k = torch.randn(1, 4, 16, 32)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
