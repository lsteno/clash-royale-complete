FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# JAX/XLA: reduce “grab all VRAM” behavior by default.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    git curl ca-certificates \
    ffmpeg \
    build-essential pkg-config \
    libleptonica-dev libtesseract-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools wheel

WORKDIR /app

# Copy sources.
COPY dreamerv3-main dreamerv3-main
COPY clash-royale-mbrl clash-royale-mbrl
COPY KataCR KataCR

# Install DreamerV3 + core deps (elements, embodied, etc.).
RUN pip install -e dreamerv3-main

# Install JAX with CUDA 12 support (GPU training).
RUN pip install -U "jax[cuda12]==0.4.33"
# KataCR's YOLO pipeline imports Flax/Orbax utilities (even when the card
# classifier is disabled), so include them in the image.
RUN pip install -U "flax" "orbax-checkpoint"

# Install CUDA-enabled PyTorch for YOLO/Ultralytics.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.1.2+cu121" "torchvision==0.16.2+cu121"

# OCR deps (CPU by default; avoids needing Paddle GPU wheels).
RUN pip install "paddlepaddle==2.6.2" "paddleocr==2.7.0.3"
# paddleocr pulls in NumPy>=2, but Dreamer/JAX stack expects NumPy<2.
RUN pip install -U "numpy<2"

# Install Clash Royale pipeline (perception + gRPC + scripts).
RUN pip install -e clash-royale-mbrl

# Resolve version conflicts introduced by vision/OCR dependencies.
# - Dreamer/JAX stack expects NumPy < 2.
# - KataCR uses PaddleOCR, which expects OpenCV <= 4.6.0.66.
RUN pip uninstall -y opencv-python-headless opencv-contrib-python opencv-python || true \
  && pip install -U --force-reinstall "numpy<2" "opencv-python==4.6.0.66"

EXPOSE 50051

# Default: start the DreamerV3 training server (Machine A).
CMD ["python", "clash-royale-mbrl/train_dreamerv3.py", "--rpc-host", "0.0.0.0", "--rpc-port", "50051"]
