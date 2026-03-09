# MiniGPT-Forge Docker Image
# Author: Prajit Datta (https://github.com/prajitdatta)

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Prajit Datta <https://github.com/prajitdatta>"
LABEL description="MiniGPT-Forge: Build, Train & Deploy GPT Models"
LABEL url="https://github.com/prajitdatta/MiniGPT-Forge"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install API dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Default command
CMD ["python", "-c", "from minigpt import MiniGPT; print('MiniGPT-Forge ready!')"]
