"""
FastAPI Inference Server
========================

Production-ready REST API for text generation with
streaming support via Server-Sent Events (SSE).

Usage:
    python -m minigpt.api.server --checkpoint path/to/model.pt --port 8000

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import os
import time
import uuid
import argparse
from typing import Optional

import torch

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI and uvicorn are required for the API server. "
        "Install with: pip install fastapi uvicorn"
    )

from minigpt.model.gpt import MiniGPT
from minigpt.model.generation import generate, GenerationConfig
from minigpt.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    UsageInfo,
    ModelInfo,
)

# ============================================================
# App setup
# ============================================================

app = FastAPI(
    title="MiniGPT-Forge API",
    description="Text generation API powered by MiniGPT-Forge",
    version="0.4.2",
    docs_url="/docs",
    contact={
        "name": "Prajit Datta",
        "url": "https://prajitdatta.github.io/",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
_model: Optional[MiniGPT] = None
_tokenizer = None
_device = None
_model_name = "minigpt-forge"


def load_model(checkpoint_path: str, device: str = "auto"):
    """Load model from checkpoint."""
    global _model, _tokenizer, _device, _model_name

    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    _model = MiniGPT.from_pretrained(checkpoint_path, device=str(_device))
    _model_name = f"minigpt-forge-{_model.num_parameters() // 1_000_000}M"

    # Load tiktoken tokenizer
    try:
        from minigpt.data.tokenizer import TiktokenWrapper
        _tokenizer = TiktokenWrapper("gpt2")
    except Exception:
        print("Warning: tiktoken not available, using basic tokenizer")

    print(f"Model loaded: {_model_name} on {_device}")


# ============================================================
# Endpoints
# ============================================================


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": _model_name}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            ModelInfo(
                id=_model_name,
                object="model",
                owned_by="prajitdatta",
                parameters=_model.num_parameters() if _model else 0,
            ).dict()
        ]
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Generate text completion."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize input
        input_ids = torch.tensor(
            [_tokenizer.encode(request.prompt)],
            dtype=torch.long,
            device=_device,
        )

        prompt_tokens = input_ids.size(1)

        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            seed=request.seed,
        )

        t0 = time.time()
        output_ids = generate(
            model=_model,
            input_ids=input_ids,
            config=config,
        )
        elapsed = time.time() - t0

        # Decode only the generated portion
        generated_ids = output_ids[0, prompt_tokens:].tolist()
        text = _tokenizer.decode(generated_ids)

        completion_tokens = len(generated_ids)

        return CompletionResponse(
            id=f"gen-{uuid.uuid4().hex[:8]}",
            model=_model_name,
            choices=[
                CompletionChoice(
                    text=text,
                    finish_reason="length" if completion_tokens >= request.max_tokens else "stop",
                    tokens_generated=completion_tokens,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions/stream")
async def create_completion_stream(request: CompletionRequest):
    """Streaming text generation via Server-Sent Events."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def event_stream():
        input_ids = torch.tensor(
            [_tokenizer.encode(request.prompt)],
            dtype=torch.long,
            device=_device,
        )

        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )

        _model.eval()
        _model.reset_caches()
        generated = input_ids.clone()
        block_size = _model.block_size

        for step in range(config.max_tokens):
            context = generated[:, -block_size:]

            with torch.no_grad():
                logits = _model(context)

            next_logits = logits[:, -1, :] / max(config.temperature, 1e-8)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            token_text = _tokenizer.decode([next_token.item()])
            yield f"data: {token_text}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# CLI entrypoint
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="MiniGPT-Forge API Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    load_model(args.checkpoint, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
