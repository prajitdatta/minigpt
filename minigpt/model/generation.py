"""
Text Generation Engine
======================

Implements autoregressive text generation with various
decoding strategies: greedy, top-k, top-p (nucleus), beam search,
and temperature-based sampling.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_tokens: List[int] = field(default_factory=list)
    num_beams: int = 1
    do_sample: bool = True
    seed: Optional[int] = None


def top_k_filtering(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Filter logits to keep only the top-k highest values.

    Args:
        logits: Raw logits of shape (batch_size, vocab_size).
        top_k: Number of top values to keep.

    Returns:
        Filtered logits with non-top-k values set to -inf.
    """
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_value, float("-inf"))


def top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """
    Nucleus (top-p) filtering — keep smallest set of tokens
    whose cumulative probability exceeds top_p.

    Args:
        logits: Raw logits of shape (batch_size, vocab_size).
        top_p: Cumulative probability threshold.

    Returns:
        Filtered logits.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order
    return sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeated tokens.

    Args:
        logits: Raw logits of shape (batch_size, vocab_size).
        generated_ids: Previously generated token IDs.
        penalty: Penalty factor (>1.0 to penalize, 1.0 = no penalty).

    Returns:
        Penalized logits.
    """
    if penalty == 1.0:
        return logits

    for i in range(logits.size(0)):
        unique_ids = generated_ids[i].unique()
        for token_id in unique_ids:
            if logits[i, token_id] > 0:
                logits[i, token_id] /= penalty
            else:
                logits[i, token_id] *= penalty

    return logits


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    config: Optional[GenerationConfig] = None,
    tokenizer=None,
    stream_callback: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Autoregressive text generation.

    Args:
        model: The language model (must have a forward method returning logits).
        input_ids: Starting token IDs of shape (batch_size, seq_len).
        config: Generation configuration.
        tokenizer: Optional tokenizer for decoding (used with stream_callback).
        stream_callback: Optional function called with each new token for streaming.

    Returns:
        Generated token IDs of shape (batch_size, input_len + generated_len).
    """
    if config is None:
        config = GenerationConfig()

    device = input_ids.device
    model.eval()

    # Reset KV caches
    if hasattr(model, "reset_caches"):
        model.reset_caches()

    # Set seed for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)

    generated = input_ids.clone()
    block_size = getattr(model, "block_size", 1024)

    for step in range(config.max_tokens):
        # Crop context to block_size
        context = generated[:, -block_size:]

        # Forward pass
        logits = model(context)

        # Get logits for the last position
        next_logits = logits[:, -1, :]

        # Apply temperature
        if config.temperature != 1.0:
            next_logits = next_logits / config.temperature

        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(
                next_logits, generated, config.repetition_penalty
            )

        # Apply frequency and presence penalties
        if config.frequency_penalty != 0.0 or config.presence_penalty != 0.0:
            for i in range(generated.size(0)):
                token_counts = torch.bincount(
                    generated[i], minlength=next_logits.size(-1)
                ).float()
                next_logits[i] -= config.frequency_penalty * token_counts
                next_logits[i] -= config.presence_penalty * (token_counts > 0).float()

        # Apply top-k filtering
        if config.top_k > 0:
            next_logits = top_k_filtering(next_logits, config.top_k)

        # Apply top-p (nucleus) filtering
        if config.top_p < 1.0:
            next_logits = top_p_filtering(next_logits, config.top_p)

        # Sample or greedy decode
        if config.do_sample:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Stream callback
        if stream_callback is not None:
            stream_callback(next_token, step)

        # Check for stop tokens
        if config.stop_tokens:
            if next_token.item() in config.stop_tokens:
                break

    return generated
