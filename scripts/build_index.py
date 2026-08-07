"""
build_index.py
--------------
Chunks scraped Merck Manual text, embeds it via Anaconda Desktop's local
model server, and builds:
  - data/index/merck.faiss      — FAISS dense vector index
  - data/index/chunks.json      — chunk metadata (text, source, section, slug)

Usage:
    conda activate anaconda-clinical-rag
    python scripts/build_index.py

Requires:
    - data/raw/ populated first (run scripts/scraper.py)
    - Anaconda Desktop running with Qwen3-Embedding-8B loaded as a model server
      (default: http://localhost:8080)

Chunks scraped text, embeds each chunk via the local /v1/embeddings endpoint,
and writes a FAISS index.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import faiss
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR       = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
INDEX_DIR     = Path("data/index")
FAISS_PATH    = INDEX_DIR / "merck.faiss"
CHUNKS_PATH   = INDEX_DIR / "chunks.json"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# Embedding server (indexing only needs the embedding model). Prefer the
# role-specific EMBEDDING_API_URL — with `anaconda ai launch` each model gets its
# own random port (written to .env by serve_models.sh), so DESKTOP_API_URL's
# :8080 default is just a last-resort fallback.
DESKTOP_API_BASE = (
    os.getenv("EMBEDDING_API_URL")
    or os.getenv("DESKTOP_API_URL", "http://localhost:8080/v1")
)
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")

# Chunking — word-based, no tokenizer dependency.
# Medical text tokenizes at ~1.4 tokens/word, so 200 words ≈ 280 tokens, keeping
# each request under the embedding server's 512-token batch limit. Larger chunks
# and multi-chunk batches can crash the server — see Known issues in the README.
CHUNK_SIZE_WORDS    = 200
CHUNK_OVERLAP_WORDS = 30

# One chunk per /v1/embeddings request. Batching multiple chunks per request
# has been observed to crash the server mid-run.
BATCH_SIZE = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, slug: str, section: str, url: str) -> list[dict]:
    """
    Split text into overlapping word-bounded chunks.

    Word-based chunking removes the tokenizer dependency entirely.

    Returns list of chunk dicts with text + metadata.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))
        chunk_str = " ".join(words[start:end]).strip()

        # Skip near-empty chunks
        if len(chunk_str) > 50:
            chunks.append({
                "chunk_id":    f"{slug}_{chunk_idx:04d}",
                "slug":        slug,
                "section":     section,
                "url":         url,
                "chunk_idx":   chunk_idx,
                "text":        chunk_str,
                "word_count":  end - start,
            })
            chunk_idx += 1

        start += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    return chunks


# ── Embedding via Anaconda Desktop API ────────────────────────────────────────

def embed_texts(texts: list[str], api_url: str, model: str) -> np.ndarray:
    """
    Embed a list of texts by calling the local /v1/embeddings endpoint.

    For Qwen3-Embedding document embeddings no instruction prefix is used
    (asymmetric design: instruction on query side only).

    Returns float32 array of shape (n, embedding_dim), L2-normalised.
    """
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        # Retry up to 3 times — Desktop server can briefly disconnect
        # after saving prompt cache state between requests.
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{api_url}/embeddings",
                    json={"model": model, "input": batch},
                    timeout=120,
                )
                resp.raise_for_status()
                break  # success
            except requests.exceptions.ConnectionError as e:
                if attempt < 2:
                    log.warning(f"Connection dropped on chunk {i}, retry {attempt + 1}/3 ...")
                    time.sleep(2)
                else:
                    log.error(
                        f"Cannot reach Anaconda Desktop at {api_url} after 3 attempts. "
                        "Is the model server running? "
                        "In Desktop: select Qwen3-Embedding-8B → Start Server."
                    )
                    raise
            except requests.exceptions.HTTPError as e:
                log.error(f"Desktop API error: {e} — {resp.text[:200]}")
                raise

        data = resp.json()
        # OpenAI-compatible response: data.data is a list sorted by index
        batch_vecs = [
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]
        all_embeddings.extend(batch_vecs)

    arr = np.array(all_embeddings, dtype="float32")

    # L2-normalise so IndexFlatIP gives cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr = arr / norms

    return arr


# ── Server readiness check ────────────────────────────────────────────────────

def wait_for_server(api_url: str, timeout: int = 60) -> None:
    """
    Poll the Desktop server until it responds, or timeout.
    Prevents sending embedding requests before the server is fully loaded.
    """
    log.info(f"Waiting for Desktop server at {api_url} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{api_url}/models", timeout=5)
            if resp.status_code == 200:
                log.info("Server is ready.")
                return
        except Exception:
            pass
        log.info("  Server not ready yet, retrying in 3s ...")
        time.sleep(3)
    raise TimeoutError(
        f"Desktop server at {api_url} did not become ready within {timeout}s. "
        "Is Qwen3-Embedding-8B running in Anaconda Desktop?"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest for section metadata
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
    log.info(f"Embedding model: {EMBEDDING_MODEL} via {DESKTOP_API_BASE}")

    # ── Chunking ──────────────────────────────────────────────────────────────
    all_chunks = []

    for path in tqdm(raw_files, desc="Chunking"):
        slug = path.stem
        meta = manifest_map.get(slug, {})
        section = meta.get("section", "Unknown")
        url = meta.get("url", "")

        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, slug, section, url)
        all_chunks.extend(chunks)
        log.info(f"  {slug}: {len(chunks)} chunks")

    log.info(f"\nTotal chunks: {len(all_chunks)}")

    # ── Embed via Anaconda Desktop ────────────────────────────────────────────
    # Wait for Desktop server to be ready before sending embedding requests.
    # Prevents ConnectionRefusedError when server is still loading after a restart.
    wait_for_server(DESKTOP_API_BASE)

    log.info(f"Embedding {len(all_chunks)} chunks via Anaconda Desktop...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, DESKTOP_API_BASE, EMBEDDING_MODEL)

    # ── FAISS index ───────────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    log.info(f"Building FAISS index (dim={dim}, n={len(all_chunks)})...")

    index = faiss.IndexFlatIP(dim)  # inner product on normalised vecs = cosine
    index.add(embeddings)

    # Write vectors and metadata together, after embedding succeeds, so a failure
    # mid-run cannot leave chunks.json describing a different set than the index.
    faiss.write_index(index, str(FAISS_PATH))
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    log.info(f"FAISS index saved → {FAISS_PATH}")
    log.info(f"Chunk metadata saved → {CHUNKS_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n── Index build complete ──────────────────────────────")
    log.info(f"  Topics:    {len(raw_files)}")
    log.info(f"  Chunks:    {len(all_chunks)}")
    log.info(f"  Embedding: {EMBEDDING_MODEL} (dim={dim}) — via Anaconda Desktop")
    log.info(f"  FAISS:     {FAISS_PATH}")
    log.info(f"  Metadata:  {CHUNKS_PATH}")
    log.info("─────────────────────────────────────────────────────\n")
    log.info("Next: uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
