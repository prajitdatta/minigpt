# Deployment Guide

<p align="center">
  <img src="https://static.wixstatic.com/media/68ad1b_96c952149d584505bdfc30a3cbf36795~mv2.jpg" width="60" alt="MiniGPT-Forge"/>
</p>

## Docker Deployment

```bash
docker build -t minigpt-forge .
docker run --gpus all -p 8000:8000 minigpt-forge \
  python -m minigpt.api.server --checkpoint /app/checkpoints/model.pt
```

## Cloud Deployment

### AWS EC2
Recommended: `g5.xlarge` (A10G GPU) for models up to 774M parameters.

### Google Cloud
Recommended: `a2-highgpu-1g` (A100 GPU) for 1.5B+ parameter models.

## Production Considerations

- Enable `torch.compile` for 2x inference speedup
- Use BF16 precision for Ampere+ GPUs
- Set `OMP_NUM_THREADS=1` for single-request latency optimization
- Consider ONNX export for maximum throughput

---

<p align="center">
  <a href="https://github.com/prajitdatta/MiniGPT-Forge">Back to Repository</a> •
  Built by <a href="https://github.com/prajitdatta">Prajit Datta</a>
</p>
