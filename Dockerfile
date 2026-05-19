# Dockerfile
# ----------
# Base image for the Clinical Knowledge API on Outerbounds.
# Uses Miniconda on NVIDIA CUDA — all Python packages installed via
# conda from the Anaconda main channel. No pip.
#
# Build:
#   docker build -t anaconda/vllm-clinical-rag:latest .
#
# Note: ana ob deploy handles building and pushing this image automatically.
# The image tag must match the reference in ob_app.py.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System deps + Miniconda ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# ── Create conda environment from project spec ────────────────────────────────
# All packages installed from Anaconda main channel.
# vllm GPU is the only pip fallback — conda GPU build in progress.
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda clean -afy

# Make the environment active for all subsequent commands
SHELL ["conda", "run", "-n", "vllm-clinical-rag", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/vllm-clinical-rag/bin:$PATH

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/        ./src/
COPY scripts/    ./scripts/
COPY eval/       ./eval/
COPY ob_app.py   ./ob_app.py
COPY .env.example .env.example

# Data indexes are generated at runtime — not baked into the image.
# Run scripts/scraper.py + scripts/build_index.py once, then mount
# data/index/ via Outerbounds storage or bake in with:
#   COPY data/index/ ./data/index/

ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/hf_cache

EXPOSE 8000 8001

CMD ["conda", "run", "--no-capture-output", "-n", "vllm-clinical-rag", \
     "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
