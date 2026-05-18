"""
api.py
------
FastAPI Clinical Knowledge API backed by vLLM.

Endpoints:
    POST /query          — answer a clinical question
    GET  /health         — liveness check
    GET  /docs           — auto-generated Swagger UI (FastAPI default)

Start:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.retriever import HybridRetriever
from src.vllm_client import VLLMClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Startup / shutdown ────────────────────────────────────────────────────────

retriever: HybridRetriever | None = None
llm_client: VLLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_client
    log.info("Loading retriever and vLLM client...")
    retriever = HybridRetriever(
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/index/merck.faiss"),
        bm25_path="data/index/bm25.pkl",
        chunks_path=os.getenv("CHUNK_METADATA_PATH", "data/index/chunks.json"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "thenlper/gte-large"),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", 10)),
        top_k_rerank=int(os.getenv("TOP_K_RERANK", 4)),
    )
    llm_client = VLLMClient(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1"),
        model=os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
    )
    log.info("API ready.")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical Knowledge API",
    description=(
        "Answers clinical queries grounded in the Merck Manual Professional Edition. "
        "Powered by vLLM inference, FAISS + BM25 hybrid retrieval, and cross-encoder re-ranking. "
        "Built with Anaconda CLI + Outerbounds. For educational and demonstration purposes only."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        example="What is the protocol for managing sepsis in a critical care unit?",
    )
    max_tokens: Optional[int] = Field(default=512, ge=64, le=1024)
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=1.0)


class SourceRef(BaseModel):
    slug: str
    section: str
    url: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]
    model: str
    latency_ms: int
    usage: dict


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int
    model: str
    vllm_url: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — confirms retriever and vLLM client are loaded."""
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {
        "status": "ok",
        "chunks_loaded": len(retriever.chunks),
        "model": llm_client.model,
        "vllm_url": llm_client.base_url,
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Answer a clinical question grounded in the Merck Manual.

    The response includes:
    - A structured clinical answer with citations
    - Source references (section, slug, URL)
    - A mandatory medical disclaimer
    - Token usage and latency
    """
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    t0 = time.perf_counter()

    # Retrieve relevant chunks
    try:
        chunks = retriever.retrieve(request.question)
    except Exception as e:
        log.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    # Generate answer
    try:
        result = llm_client.generate(
            question=request.question,
            chunks=chunks,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(status_code=502, detail=f"Inference failed: {e}")

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
        model=result["model"],
        latency_ms=latency_ms,
        usage=result["usage"],
    )
