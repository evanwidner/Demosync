FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    git curl wget ca-certificates \
    build-essential cmake ninja-build \
    ffmpeg \
    colmap \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip wheel setuptools

WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# tinycudann (required by nerfstudio for hashgrid features). Pinned to v2.0
# (commit 3c7931c, 2025-07-08) so CI builds are reproducible — bump deliberately.
RUN pip install ninja \
    git+https://github.com/NVlabs/tiny-cuda-nn.git@3c7931caab7b691c6f30dffe20952ed727d935b9#subdirectory=bindings/torch

COPY . /app/worker

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "worker.pipeline"]
