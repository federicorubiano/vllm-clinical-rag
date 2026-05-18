# Dockerfile
# ----------
# Base image for the Clinical Knowledge API on Outerbounds.
# Built on NVIDIA CUDA to support vLLM GPU inference.
#
# Build:
#   docker build -t anaconda/vllm-clinical-rag:latest .
#
# Note: ana ob deploy handles building and pushing this image automatically
# when you have Docker configured. The image tag must match ob_app.py.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

WORKDIR /app

# Python deps — install before copying app code for layer caching
COPY environment.yml .
RUN pip install --no-cache-dir \
    vllm \
    faiss-gpu \
    sentence-transformers \
    rank-bm25 \
    fastapi \
    "uvicorn[standard]" \
    openai \
    python-dotenv \
    requests \
    tiktoken \
    beautifulsoup4 \
    evidently \
    metaflow \
    outerbounds

# Copy application code
COPY src/        ./src/
COPY scripts/    ./scripts/
COPY eval/       ./eval/
COPY .env.example .env.example

# Data indexes are mounted at runtime from Outerbounds storage
# or pre-built and baked in. To bake in:
#   COPY data/index/ ./data/index/
# To mount at runtime, set FAISS_INDEX_PATH + CHUNK_METADATA_PATH in .env

ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/hf_cache

EXPOSE 8000 8001

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
