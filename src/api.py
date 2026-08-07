"""
api.py
------
FastAPI Clinical Knowledge API backed by Anaconda Desktop local inference.

Endpoints:
    POST /query          — answer a clinical question
    GET  /health         — liveness check
    GET  /docs           — auto-generated Swagger UI (FastAPI default)

Start:
    uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
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
from src.desktop_client import DesktopClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Startup / shutdown ────────────────────────────────────────────────────────

retriever: HybridRetriever | None = None
llm_client: DesktopClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_client
    log.info("Loading retriever and Anaconda Desktop client...")
    # Use separate URLs for embedding and inference so two Desktop servers
    # (on different ports) can be configured independently via .env.
    # Fall back to DESKTOP_API_URL if the role-specific vars are not set.
    _desktop = os.getenv("DESKTOP_API_URL", "http://localhost:8080/v1")
    retriever = HybridRetriever(
        faiss_path=os.getenv("FAISS_INDEX_PATH", "data/index/merck.faiss"),
        chunks_path=os.getenv("CHUNK_METADATA_PATH", "data/index/chunks.json"),
        api_url=os.getenv("EMBEDDING_API_URL", _desktop),
        embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B"),
        top_k=int(os.getenv("TOP_K", 5)),
    )
    llm_client = DesktopClient(
        base_url=os.getenv("INFERENCE_API_URL", _desktop),
        model=os.getenv("INFERENCE_MODEL", "Qwen2.5-14B-Instruct"),
    )
    log.info("API ready.")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical Knowledge API",
    description=(
        "Answers clinical queries grounded in the Merck Manual Professional Edition. "
        "Powered by Anaconda Desktop local inference (Qwen2.5-14B-Instruct), FAISS dense retrieval, "
        "and Qwen3-Embedding-8B instruction-following embeddings. "
        "Fully self-hosted — all inference runs locally through Anaconda Desktop. "
        "Built with Anaconda CLI + Anaconda Desktop. For educational and demonstration purposes only."
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
    desktop_url: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — confirms retriever and Desktop client are loaded."""
    if retriever is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {
        "status": "ok",
        "chunks_loaded": len(retriever.chunks),
        "model": llm_client.model,
        "desktop_url": llm_client.base_url,
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
