"""
retriever.py
------------
Dense retrieval pipeline backed by Anaconda Desktop's local model server.

  1. FAISS dense search  — semantic similarity via Qwen3-Embedding embeddings
                           (query embedded with clinical instruction prefix)
  2. Top-k returned      — ordered by cosine similarity from the FAISS search

All embedding calls go to Anaconda Desktop's OpenAI-compatible /v1/embeddings.
"""

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import faiss
import requests

log = logging.getLogger(__name__)

# Instruction prefix for query-side embedding (Qwen3-Embedding asymmetric design).
# Documents are embedded without a prefix; queries use this instruction so the
# model understands retrieval intent. This is the standard Qwen3-Embedding usage.
QUERY_INSTRUCTION = (
    "Instruct: Given a clinical question, retrieve relevant medical passages "
    "that answer the question\nQuery: "
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str
    slug: str
    section: str
    url: str
    text: str
    score: float


# ── Retriever ─────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    FAISS dense retriever using Anaconda Desktop's local embedding API.

    Embeds the query via the local /embeddings endpoint, then searches the FAISS index.

    Parameters
    ----------
    faiss_path       : path to FAISS index file
    chunks_path      : path to chunk metadata JSON
    api_url          : Anaconda Desktop model server base URL
    embedding_model  : embedding model name as shown in Desktop catalog
    top_k            : number of chunks to return
    """

    def __init__(
        self,
        faiss_path: str = "data/index/merck.faiss",
        chunks_path: str = "data/index/chunks.json",
        api_url: str | None = None,
        embedding_model: str | None = None,
        top_k: int = 5,
    ):
        self.api_url = (
            api_url or os.getenv("DESKTOP_API_URL", "http://localhost:8080/v1")
        ).rstrip("/")
        self.embedding_model = (
            embedding_model or os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
        )
        self.top_k = top_k

        log.info("Loading chunk metadata...")
        self.chunks: list[dict] = json.loads(Path(chunks_path).read_text())

        log.info("Loading FAISS index...")
        self.faiss_index = faiss.read_index(faiss_path)

        log.info(
            f"Retriever ready — {len(self.chunks)} chunks indexed. "
            f"Embedding: {self.embedding_model} via {self.api_url}"
        )

    # ── Query embedding ───────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query using the clinical instruction prefix.

        Qwen3-Embedding is an instruction-following model: queries include
        a task instruction, documents do not.
        """
        instructed = QUERY_INSTRUCTION + query

        try:
            resp = requests.post(
                f"{self.api_url}/embeddings",
                json={"model": self.embedding_model, "input": [instructed]},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            log.error(
                f"Cannot reach Anaconda Desktop at {self.api_url}. "
                "Is the embedding model server running?"
            )
            raise

        data = resp.json()
        vec = np.array(data["data"][0]["embedding"], dtype="float32")

        # L2-normalise (index was built with normalised vectors)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.reshape(1, -1)

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Embed the query and search FAISS.

        Returns top-k RetrievedChunk objects ordered by cosine similarity.
        """
        query_vec = self._embed_query(query)
        scores, indices = self.faiss_index.search(query_vec, self.top_k)

        results = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if idx < 0:
                continue  # FAISS returns -1 for empty slots
            c = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    slug=c["slug"],
                    section=c["section"],
                    url=c["url"],
                    text=c["text"],
                    score=float(score),
                )
            )

        return results


# Backwards-compatible alias
HybridRetriever = DenseRetriever
