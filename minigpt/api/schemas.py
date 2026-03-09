"""
API Request/Response Schemas
=============================

Pydantic models for the MiniGPT-Forge REST API.

Author: Prajit Datta (https://github.com/prajitdatta)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Request body for text completion."""

    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(128, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, description="Top-k filtering (0 = disabled)")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="Repetition penalty")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    stream: bool = Field(False, description="Enable streaming response")


class CompletionChoice(BaseModel):
    """A single completion choice."""

    text: str
    finish_reason: str  # "stop" or "length"
    tokens_generated: int


class UsageInfo(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Response body for text completion."""

    id: str
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: str = "model"
    owned_by: str = "prajitdatta"
    parameters: int = 0
