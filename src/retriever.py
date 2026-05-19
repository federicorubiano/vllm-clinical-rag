"""
retriever.py
------------
Dense retrieval pipeline:
  1. FAISS dense search         — semantic similarity via gte-large embeddings
  2. Cross-encoder re-ranking   — reorder candidates by true relevance

Returns the top-k most relevant chunks for a given query,
each with source metadata for citation enforcement.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

log = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────────────

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
    FAISS dense retriever with cross-encoder re-ranking.

    Parameters
    ----------
    faiss_path       : path to FAISS index file
    chunks_path      : path to chunk metadata JSON
    embedding_model  : sentence-transformer model name
    reranker_model   : cross-encoder model name
    top_k_retrieve   : candidates to retrieve from FAISS before re-ranking
    top_k_rerank     : final chunks returned after re-ranking
    """

    def __init__(
        self,
        faiss_path: str = "data/index/merck.faiss",
        chunks_path: str = "data/index/chunks.json",
        embedding_model: str = "thenlper/gte-large",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_retrieve: int = 20,
        top_k_rerank: int = 4,
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        log.info("Loading chunk metadata...")
        self.chunks: list[dict] = json.loads(Path(chunks_path).read_text())

        log.info("Loading FAISS index...")
        self.faiss_index = faiss.read_index(faiss_path)

        log.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        log.info(f"Loading cross-encoder: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        log.info(f"Retriever ready — {len(self.chunks)} chunks indexed.")

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _dense_search(self, query: str) -> list[tuple[int, float]]:
        """Return (chunk_idx, score) pairs from FAISS."""
        vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        scores, indices = self.faiss_index.search(vec, self.top_k_retrieve)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    # ── Cross-encoder re-ranking ──────────────────────────────────────────────

    def _rerank(self, query: str, candidate_indices: list[int]) -> list[tuple[int, float]]:
        """Re-rank candidates with a cross-encoder. Returns (idx, score) pairs."""
        pairs = [(query, self.chunks[i]["text"]) for i in candidate_indices]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(candidate_indices, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[: self.top_k_rerank]

    # ── Public interface ──────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Dense retrieval pipeline.
        Returns top-k RetrievedChunk objects, best first.
        """
        dense_results = self._dense_search(query)
        candidate_indices = [idx for idx, _ in dense_results]
        ranked = self._rerank(query, candidate_indices)

        results = []
        for idx, score in ranked:
            c = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    slug=c["slug"],
                    section=c["section"],
                    url=c["url"],
                    text=c["text"],
                    score=score,
                )
            )

        return results


# Backwards-compatible alias
HybridRetriever = DenseRetriever
