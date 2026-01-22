FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# JAX/XLA: reduce “grab all VRAM” behavior by default.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git curl ca-certificates \
    ffmpeg \
    build-essential pkg-config \
    libleptonica-dev libtesseract-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools wheel

WORKDIR /app

# Copy only manifests first for caching.
COPY dreamerv3-main/requirements.txt dreamerv3-main/requirements.txt
COPY dreamerv3-main/setup.py dreamerv3-main/setup.py
COPY dreamerv3-main/README.md dreamerv3-main/README.md
COPY clash-royale-mbrl/pyproject.toml clash-royale-mbrl/pyproject.toml

# Install DreamerV3 + core deps (JAX, elements, embodied, etc.).
# Note: This installs DreamerV3 requirements, not the heavy optional envs from its Dockerfile.
RUN pip install -e dreamerv3-main

# Install CUDA-enabled PyTorch for YOLO/Ultralytics.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.1.2+cu121" "torchvision==0.16.2+cu121"

# OCR deps (CPU by default; avoids needing Paddle GPU wheels).
RUN pip install "paddlepaddle==2.6.2" "paddleocr==2.7.0.3"

# Install Clash Royale pipeline (perception + gRPC + scripts).
COPY clash-royale-mbrl clash-royale-mbrl
RUN pip install -e clash-royale-mbrl

# The KataCR repo is copied to make imports work, but weights should be mounted at runtime.
COPY KataCR KataCR

EXPOSE 50051

# Default: start the DreamerV3 training server (Machine A).
CMD ["python", "clash-royale-mbrl/train_dreamerv3.py", "--rpc-host", "0.0.0.0", "--rpc-port", "50051"]
