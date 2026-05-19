"""
build_index.py
--------------
Chunks scraped Merck Manual text, embeds it, and builds:
  - data/index/merck.faiss      — FAISS dense vector index
  - data/index/bm25.pkl         — BM25 sparse index
  - data/index/chunks.json      — chunk metadata (text, source, section, slug)

Usage:
    conda activate vllm-clinical-rag
    python scripts/build_index.py

Requires data/raw/ to be populated first (run scripts/scraper.py).

All packages from Anaconda main channel — no pip dependencies.
"""

import json
import pickle
import logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw")
INDEX_DIR     = Path("data/index")
FAISS_PATH    = INDEX_DIR / "merck.faiss"
BM25_PATH     = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH   = INDEX_DIR / "chunks.json"
MANIFEST_PATH = RAW_DIR / "manifest.json"

EMBEDDING_MODEL  = "thenlper/gte-large"
CHUNK_SIZE       = 400   # tokens — stays within gte-large's 512-token limit
CHUNK_OVERLAP    = 50    # tokens — enough to keep clinical concepts intact
BATCH_SIZE       = 32    # embedding batch size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, slug: str, section: str, url: str, tokenizer) -> list[dict]:
    """
    Split text into overlapping token-bounded chunks using the embedding
    model's own tokenizer (from transformers, via sentence-transformers).
    Returns list of chunk dicts with text + metadata.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(token_ids):
        end = min(start + CHUNK_SIZE, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_str = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        # Skip near-empty chunks
        if len(chunk_str) > 50:
            chunks.append({
                "chunk_id":    f"{slug}_{chunk_idx:04d}",
                "slug":        slug,
                "section":     section,
                "url":         url,
                "chunk_idx":   chunk_idx,
                "text":        chunk_str,
                "token_count": len(chunk_ids),
            })
            chunk_idx += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Embed all chunks in batches. Returns float32 array (n_chunks, dim)."""
    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via inner product
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest to get section metadata
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"No manifest found at {MANIFEST_PATH}. "
            "Run scripts/scraper.py first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_map = {m["slug"]: m for m in manifest if m.get("status") in ("ok", "cached")}

    # Collect raw text files
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(
            f"No .txt files found in {RAW_DIR}. "
            "Run scripts/scraper.py first."
        )

    log.info(f"Found {len(raw_files)} raw topic files.")

    # ── Load tokenizer (same model used for embeddings) ───────────────────────
    log.info(f"Loading tokenizer: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    # ── Chunking ─────────────────────────────────────────────────────────────
    all_chunks = []

    for path in tqdm(raw_files, desc="Chunking"):
        slug = path.stem
        meta = manifest_map.get(slug, {})
        section = meta.get("section", "Unknown")
        url = meta.get("url", "")

        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, slug, section, url, tokenizer)
        all_chunks.extend(chunks)
        log.info(f"  {slug}: {len(chunks)} chunks")

    log.info(f"\nTotal chunks: {len(all_chunks)}")

    # ── Save chunk metadata ───────────────────────────────────────────────────
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    log.info(f"Chunk metadata saved → {CHUNKS_PATH}")

    # ── BM25 index ───────────────────────────────────────────────────────────
    log.info("Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    log.info(f"BM25 index saved → {BM25_PATH}")

    # ── FAISS index ───────────────────────────────────────────────────────────
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = embed_chunks(all_chunks, model)
    dim = embeddings.shape[1]

    log.info(f"Building FAISS index (dim={dim}, n={len(all_chunks)})...")
    index = faiss.IndexFlatIP(dim)  # inner product = cosine (normalized vecs)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_PATH))
    log.info(f"FAISS index saved → {FAISS_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n── Index build complete ──────────────────────────────")
    log.info(f"  Topics:    {len(raw_files)}")
    log.info(f"  Chunks:    {len(all_chunks)}")
    log.info(f"  Embedding: {EMBEDDING_MODEL} (dim={dim})")
    log.info(f"  FAISS:     {FAISS_PATH}")
    log.info(f"  BM25:      {BM25_PATH}")
    log.info(f"  Metadata:  {CHUNKS_PATH}")
    log.info("─────────────────────────────────────────────────────\n")
    log.info("Next: start the vLLM server, then run: uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
