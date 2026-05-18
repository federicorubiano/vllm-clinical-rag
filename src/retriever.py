"""
retriever.py
------------
Hybrid retrieval pipeline:
  1. BM25 sparse search         — keyword precision
  2. FAISS dense search         — semantic similarity
  3. Reciprocal Rank Fusion     — merge the two result sets
  4. Cross-encoder re-ranking   — reorder by true relevance

Returns the top-k most relevant chunks for a given query,
each with source metadata for citation enforcement.
"""

import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
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

class HybridRetriever:
    """
    Hybrid BM25 + FAISS retriever with cross-encoder re-ranking.

    Parameters
    ----------
    faiss_path       : path to FAISS index file
    bm25_path        : path to pickled BM25 index
    chunks_path      : path to chunk metadata JSON
    embedding_model  : sentence-transformer model name
    reranker_model   : cross-encoder model name
    top_k_retrieve   : candidates to retrieve before re-ranking (per method)
    top_k_rerank     : final chunks returned after re-ranking
    rrf_k            : Reciprocal Rank Fusion constant (default 60)
    """

    def __init__(
        self,
        faiss_path: str = "data/index/merck.faiss",
        bm25_path: str = "data/index/bm25.pkl",
        chunks_path: str = "data/index/chunks.json",
        embedding_model: str = "thenlper/gte-large",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_retrieve: int = 10,
        top_k_rerank: int = 4,
        rrf_k: int = 60,
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.rrf_k = rrf_k

        log.info("Loading chunk metadata...")
        self.chunks: list[dict] = json.loads(Path(chunks_path).read_text())

        log.info("Loading FAISS index...")
        self.faiss_index = faiss.read_index(faiss_path)

        log.info("Loading BM25 index...")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

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

    # ── Sparse retrieval ──────────────────────────────────────────────────────

    def _sparse_search(self, query: str) -> list[tuple[int, float]]:
        """Return (chunk_idx, score) pairs from BM25."""
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)

        top_indices = np.argsort(scores)[::-1][: self.top_k_retrieve]
        return [(int(i), float(scores[i])) for i in top_indices]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    def _rrf_merge(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
    ) -> list[int]:
        """
        Merge two ranked lists via Reciprocal Rank Fusion.
        Returns candidate chunk indices ordered by fused score.
        """
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank + 1)

        merged = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return merged[: self.top_k_retrieve * 2]  # take union for re-ranking

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
        Full hybrid retrieval pipeline.
        Returns top-k RetrievedChunk objects, best first.
        """
        dense  = self._dense_search(query)
        sparse = self._sparse_search(query)
        merged = self._rrf_merge(dense, sparse)
        ranked = self._rerank(query, merged)

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
