"""
MiniGPT Model
=============

Full GPT model implementation assembling all components:
embeddings, transformer blocks, and language model head.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from minigpt.model.embeddings import TransformerEmbedding
from minigpt.model.transformer import TransformerBlock
from minigpt.model.generation import generate, GenerationConfig
from minigpt.utils.config import ModelConfig


class MiniGPT(nn.Module):
    """
    MiniGPT: A clean, from-scratch GPT implementation.

    This is the main model class that assembles token embeddings,
    a stack of transformer blocks, and a language model head.

    Args:
        config: Model configuration object (see ModelConfig).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # ---- Embedding layers ----
        self.embeddings = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.block_size,
            dropout=config.dropout,
            pos_type=config.pos_type,
        )

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                block_size=config.block_size,
                dropout=config.dropout,
                bias=config.bias,
                activation=config.activation,
                flash=config.flash_attention,
                layer_idx=i,
            )
            for i in range(config.n_layers)
        ])

        # ---- Final layer norm ----
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)

        # ---- Language model head ----
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ---- Weight tying ----
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.token_emb.embedding.weight

        # ---- Initialize weights ----
        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        # Report parameter count
        n_params = self.num_parameters()
        print(f"MiniGPT initialized with {n_params:,} parameters")

    def _init_weights(self, module: nn.Module):
        """Initialize weights with GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """
        Return total number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_emb.embedding.weight.numel()
            if hasattr(self.embeddings.pos_emb, "embedding"):
                n_params -= self.embeddings.pos_emb.embedding.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GPT model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            targets: Optional target IDs for loss computation.

        Returns:
            If targets is None: logits of shape (B, T, vocab_size).
            If targets is provided: tuple of (logits, loss).
        """
        B, T = input_ids.size()
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block size {self.block_size}"
        )

        # Embed tokens + positions
        x = self.embeddings(input_ids)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        input_ids: Optional[torch.Tensor] = None,
        tokenizer=None,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[list] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        High-level generation interface.

        Args:
            prompt: Text prompt to continue from.
            input_ids: Optional pre-tokenized input.
            tokenizer: Tokenizer instance for encoding/decoding.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeated tokens.
            stop_tokens: List of token IDs that stop generation.
            seed: Random seed for reproducibility.
            stream: If True, print tokens as they are generated.

        Returns:
            Generated text string.
        """
        if input_ids is None:
            assert tokenizer is not None, "Must provide either input_ids or tokenizer"
            input_ids = torch.tensor(
                [tokenizer.encode(prompt)],
                dtype=torch.long,
                device=next(self.parameters()).device,
            )

        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens or [],
            do_sample=temperature > 0,
            seed=seed,
        )

        stream_cb = None
        if stream and tokenizer:
            def stream_cb(token, step):
                print(tokenizer.decode([token.item()]), end="", flush=True)

        output_ids = generate(
            model=self,
            input_ids=input_ids,
            config=config,
            tokenizer=tokenizer,
            stream_callback=stream_cb,
        )

        if tokenizer:
            return tokenizer.decode(output_ids[0].tolist())
        return output_ids

    def reset_caches(self):
        """Clear all KV caches in transformer blocks."""
        for block in self.blocks:
            block.reset_cache()

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> "MiniGPT":
        """
        Load a model from a checkpoint.

        Args:
            checkpoint_path: Path to the saved checkpoint.
            device: Device to load the model on.

        Returns:
            Loaded MiniGPT model.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, **kwargs):
        """
        Save model checkpoint.

        Args:
            path: Save path.
            optimizer: Optional optimizer state to save.
            epoch: Current epoch number.
        """
        checkpoint = {
            "config": self.config,
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)

    def get_config_dict(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_heads": self.config.n_heads,
            "n_layers": self.config.n_layers,
            "d_ff": self.config.d_ff,
            "block_size": self.config.block_size,
            "dropout": self.config.dropout,
            "parameters": self.num_parameters(),
        }
