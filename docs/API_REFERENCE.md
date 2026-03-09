# API Reference

<p align="center">
  <img src="https://static.wixstatic.com/media/68ad1b_96c952149d584505bdfc30a3cbf36795~mv2.jpg" width="60" alt="MiniGPT-Forge"/>
</p>

## REST API

### Launch Server

```bash
python -m minigpt.api.server --checkpoint checkpoints/best_model.pt --port 8000
```

### Endpoints

#### `POST /v1/completions`

Generate text completion.

**Request Body:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Input text |
| `max_tokens` | int | 128 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0 = greedy) |
| `top_k` | int | 50 | Top-k filtering |
| `top_p` | float | 0.9 | Nucleus sampling |
| `repetition_penalty` | float | 1.0 | Penalty for repeated tokens |
| `seed` | int | null | Random seed |

#### `POST /v1/completions/stream`

Streaming text generation via Server-Sent Events. Same request body as above.

#### `GET /v1/models`

List available models.

#### `GET /health`

Health check. Returns `{"status": "healthy"}`.

## Python API

### MiniGPT

```python
from minigpt import MiniGPT, GPT2SmallConfig

model = MiniGPT(GPT2SmallConfig())
model.generate(prompt="Hello", tokenizer=tokenizer, max_tokens=100)
model.save_checkpoint("model.pt")
model = MiniGPT.from_pretrained("model.pt")
```

### Trainer

```python
from minigpt import Trainer

trainer = Trainer(model, train_loader, learning_rate=3e-4, max_epochs=10)
trainer.train()
```

### Tokenizer

```python
from minigpt.data import BPETokenizer, TiktokenWrapper

# Custom BPE
tok = BPETokenizer(vocab_size=8000)
tok.train(text)
ids = tok.encode("hello world")

# Tiktoken (recommended)
tok = TiktokenWrapper("gpt2")
ids = tok.encode("hello world")
```

---

<p align="center">
  <a href="https://github.com/prajitdatta/MiniGPT-Forge">Back to Repository</a> •
  Built by <a href="https://github.com/prajitdatta">Prajit Datta</a>
</p>
