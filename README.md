# 🏥 Clinical Knowledge RAG — Anaconda Desktop + FAISS + FastAPI

> **Owner:** Federico Rubiano ([@federicorubiano](https://github.com/federicorubiano)) | **Status:** In development (pending end-to-end test) | **Estimated time:** 60–90 minutes

A hands-on **guide**: build your own clinical-question RAG system, step by step, grounded in the **Merck Manual Professional Edition** — running entirely on your own machine. By the end you'll have a working, locally-served RAG you can extend.

Powered by Anaconda Desktop local inference (Qwen2.5-14B-Instruct) and FAISS dense retrieval with Qwen3-Embedding-8B instruction-following embeddings — all running locally on your own machine, no GPU required.

> *"If you can't reproduce it, you can't trust it. If you can't trust it, you can't ship it."*

Every model weight enters through Anaconda's curated catalog — auditable provenance from install to inference, zero HuggingFace Hub calls at runtime. This is what **secure by default** looks like end to end:
`ana login` → `conda env create` → open Anaconda Desktop → run locally

## Audience

Python developers and data scientists who want to build a retrieval-augmented generation (RAG) system end to end and keep every part of it running locally. No prior RAG experience required. The techniques apply to any domain that needs grounded, citable answers over a document corpus — demonstrated here on a clinical/healthcare knowledge base.

## What you'll learn

By the end of this guide you will be able to:

1. **Build** a FAISS dense-vector index from a scraped corpus using Anaconda Desktop's local embedding server.
2. **Retrieve** the most relevant passages for a question with instruction-following embeddings.
3. **Generate** grounded, citation-enforced answers from a locally-served Qwen2.5-14B-Instruct model.
4. **Serve** the pipeline as a FastAPI endpoint.
5. **Evaluate** answer quality with a reproducible scoring harness.

![Gradio UI screenshot](screenshots/gradio-ui.png)
<!-- TODO: run `python src/gradio_app.py`, open http://localhost:7860, ask a benchmark query, save to screenshots/gradio-ui.png -->

---

> ## ⚠️ MEDICAL DISCLAIMER
>
> **This project is for educational and demonstration purposes only.**
>
> Nothing produced by this system — including all generated text, citations, and clinical summaries — constitutes medical advice, diagnosis, or treatment. The system may produce inaccurate, incomplete, or outdated information even when citing real sources.
>
> **Always consult a qualified, licensed healthcare professional before making any clinical decision.** Do not use this tool in any real patient care setting. The authors, contributors, and Anaconda, Inc. accept no liability whatsoever for decisions made on the basis of this system's output.

---

## What is FAISS?

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. Here it powers semantic retrieval — finding chunks whose *meaning* matches the query, not just their keywords. Qwen3-Embedding-8B's instruction-following embeddings handle query/document asymmetry natively, so no separate reranking step is needed.

## What is Anaconda Desktop?

[Anaconda Desktop](https://www.anaconda.com/products/desktop) runs large language models **locally** on your own machine and exposes them through an OpenAI-compatible HTTP API at `localhost:8080`. You pick a model from Anaconda's curated catalog, click **Start Server**, and the rest of this project talks to it over plain `requests`. Both the embedding model (Qwen3-Embedding-8B) and the chat model (Qwen2.5-14B-Instruct) are served this way — model weights come from Anaconda's curated catalog, which keeps the whole pipeline reproducible and supply-chain clean.

## What is Evidently AI?

[Evidently AI](https://www.evidentlyai.com) is an open-source library for evaluating and monitoring ML models and LLM pipelines. Here the evaluation harness computes reproducible heuristic scores (no LLM-as-judge) and renders them through an Evidently **`DataSummaryPreset`** report — `python eval/run_eval.py --report` builds `eval/report.html` with per-metric statistics across the benchmark queries. Installed from the Anaconda `main` channel (added Q1 2026) — no pip required.

---

## Architecture

```
User Query
    │
    ▼
DenseRetriever
    └── Qwen3-Embedding-8B (query instruction prefix)
        → FAISS dense search → top-5 chunks
    │   [No cross-encoder — instruction-following asymmetry handles it]
    │
    ▼
DesktopClient  (Qwen2.5-14B-Instruct via Anaconda Desktop local server)
    │   System: citation rules + structure enforcement
    │   User:   context blocks + question
    ▼
Answer + Citations + Medical Disclaimer  [appended to every response]
    │
    └── FastAPI  /query  →  Gradio UI  →  Browser

Runs entirely locally:
    [Anaconda Desktop]
        ├── Qwen3-Embedding-8B server  → localhost:8080/v1/embeddings
        └── Qwen2.5-14B-Instruct server            → localhost:8080/v1/chat/completions
    [Your shell]
        └── FastAPI                    → 0.0.0.0:8000  ← the RAG API
```

---

## Prerequisites

**Knowledge prerequisites** (what you should already know — links fill the gaps):
- Comfortable running Python from the command line and editing `.py` files.
- Basic familiarity with conda environments ([conda environments guide](https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/environments.html)).
- A conceptual grasp of embeddings/vector search helps but isn't required ([what are embeddings?](https://www.anaconda.com/docs/tools/ai-navigator/tutorials/embedding-tutorial)). RAG itself is taught here from scratch.

**Installation prerequisites** (what must be installed — with links):
- [Anaconda or Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- [Anaconda CLI (`ana`)](https://anaconda.sh)
- [Anaconda Desktop](https://www.anaconda.com/products/desktop) — provides the local model servers for embeddings and inference
- An **Anaconda account** to sign in to Anaconda Desktop (organization users sign in with their assigned credentials)
- Python 3.12 (installed by the environment file)
- **~32 GB RAM recommended.** The default models (Qwen2.5-14B-Instruct + Qwen3-Embedding-8B) use roughly 22 GB while both servers run. On a smaller machine, see *Running on less RAM* below.
- **Speed expectation:** answers take roughly 1–3 minutes each — local inference on a 14B model is thorough, not instant.

**Dependencies:**
- **Anaconda Desktop** — required. The models run here; there's no cloud or remote fallback (that's the point — everything stays on your machine).
- **Merck Manual website** — used only to scrape the source text. If it's unreachable, a sample corpus in `data/raw/` lets you keep going.

> **Runs on Apple Silicon and any CPU-only machine.** The full pipeline — scraping, indexing, API, and UI — runs locally via Anaconda Desktop. No GPU required.

> **Running on less RAM (e.g., 16 GB).** The models are configurable, not hard-coded — point `INFERENCE_MODEL` and `EMBEDDING_MODEL` in `.env` at smaller models from the Anaconda Desktop catalog (run `anaconda ai models <name>` to see each model's RAM use). A lighter pairing that fits comfortably in 16 GB:
>
> | Role | Model | RAM (Q4_K_M) |
> |---|---|---|
> | Chat | `Qwen2.5-7B-Instruct` | ~5.2 GB |
> | Embedder | `Qwen3-Embedding-0.6B` | ~0.6 GB |
>
> Two caveats: (1) smaller models produce lower answer quality, and (2) **if you change the embedding model you must rebuild the index** (`python scripts/build_index.py`) — a different embedder produces a different vector dimension, so the existing index can't be reused. Swapping only the chat model needs no rebuild.
>
> *(This lighter pairing is a starting suggestion, not yet benchmarked against the full eval; the default 14B/8B profile is the tested one.)*

---

## Project structure

```
anaconda-clinical-rag/
├── src/
│   ├── api.py              # FastAPI /query + /health endpoints
│   ├── retriever.py        # FAISS dense retrieval (instruction-following embeddings, no reranker)
│   ├── desktop_client.py   # Anaconda Desktop chat client + prompt builder (uses requests)
│   └── gradio_app.py       # Gradio demo UI
├── scripts/
│   ├── scraper.py          # Merck Manual web scraper (robots.txt compliant)
│   ├── build_index.py      # Builds FAISS index + chunk metadata from scraped text
│   ├── serve_models.sh     # Launch both Anaconda Desktop model servers + wire .env
│   └── smoke_test.sh       # End-to-end, non-destructive setup verification
├── eval/
│   └── run_eval.py         # Heuristic evaluation harness (5 benchmark queries)
├── notebooks/
│   └── demo.ipynb          # End-to-end walkthrough notebook
├── data/
│   ├── raw/                # Scraped .txt files (git-ignored; reproduce via scraper)
│   └── index/              # FAISS index + chunk metadata (git-ignored; reproduce via build_index)
├── environment.yml         # Conda env — all packages from Anaconda main (faiss-cpu; runs on Mac/Linux)
├── anaconda-project.yml    # Anaconda Project commands
├── test_api.py             # Quick connectivity test — run this first
├── .env.example            # All config vars with guidance
└── README.md               # This file
```

---

## Build it yourself (step by step)

> Each step below opens with what you should already have working and ends with a **✅ Checkpoint** so you can verify success before moving on. You can stop after any checkpoint and still have learned something complete.

### Step 1: Login

*Start state: Anaconda/Miniconda and the `ana` CLI installed (see Prerequisites).*

```bash
ana login
```

**✅ Checkpoint:** `ana whoami` prints your Anaconda username.

`faiss-cpu`, `gradio`, `evidently`, `fastapi`, and all supporting packages are on the Anaconda `main` channel. The environment file handles all installation — no pip commands, no HuggingFace packages.

### Step 2: Create the environment

```bash
conda env create -f environment.yml
conda activate anaconda-clinical-rag
```

### Step 3: Configure your environment

```bash
cp .env.example .env
# Edit .env — set INFERENCE_MODEL / EMBEDDING_MODEL to match your Desktop servers
```

### Step 4: Start both model servers (chat + embedder, at once)

A live query needs **two models serving at the same time** — the embedder (to encode your question) and the chat model (to write the grounded answer), and the indexing step (Step 5) needs the embedder too. Anaconda Desktop's UI runs only one server at a time, so use the `anaconda ai` CLI, which gives each server its own port. The helper does it in one step:

```bash
bash scripts/serve_models.sh
```

This launches both models and writes their URLs into `.env` (`EMBEDDING_API_URL` / `INFERENCE_API_URL`). It defaults to the recommended 32 GB pair — **Qwen2.5-14B-Instruct** (chat) + **Qwen3-Embedding-8B** (embeddings) — and you can override with `INFER_SPEC` / `EMBED_SPEC`. Re-run it whenever you restart the servers (ports are assigned dynamically).

> Equivalent manual steps:
> ```bash
> anaconda ai launch Qwen2.5-14B-Instruct/Q4_K_M --detach
> anaconda ai launch Qwen3-Embedding-8B/Q8_0 --detach
> anaconda ai servers --json    # read each server's openai_url → put in .env
> ```

> No GPU required locally. Model weights come once from Anaconda's curated catalog — no HuggingFace Hub calls at runtime.

**✅ Checkpoint:** `anaconda ai servers` lists **both** as `running`, and `.env`'s `EMBEDDING_API_URL` / `INFERENCE_API_URL` now point at their two ports.

> ℹ️ If `anaconda ai` errors with a config/port (or `401`) message, point it at the running Desktop backend once: `anaconda ai config --backend anaconda-desktop -y`.

### Step 5: Scrape the Merck Manual and build the index

*Start state: your conda env is activated, `.env` is configured (Steps 2–3), and both model servers are running (Step 4).*

```bash
python scripts/scraper.py      # ~5 min — respects robots.txt 5s crawl delay
python scripts/build_index.py  # ~10 min — embeds all chunks via Anaconda Desktop
```

This writes to `data/raw/` and `data/index/`. No corpus file to download — anyone can reproduce it.

**✅ Checkpoint:** `data/index/merck.faiss` and `data/index/chunks.json` exist, and `build_index.py` prints a chunk count. Expected output (your numbers may vary):

```text
Scraped 7 topics → data/raw/
Chunked + embedded 156 passages
Wrote data/index/merck.faiss (156 vectors)
```

> ℹ️ Indexing uses the embedding server you started in Step 4. If you hit an `exit code 133` crash, see **Known issues** below — keep `CHUNK_SIZE_WORDS` at 200.

### Step 6: Start the API

*Start state: the inference model server is running in Anaconda Desktop and the index exists.*

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**✅ Checkpoint:** the log prints `API ready.` and `http://localhost:8000/docs` loads the Swagger UI.

### Step 7: Test your setup

*Start state: the API is running (Step 6).*

```bash
python test_api.py
```

**✅ Checkpoint:** you see green checkmarks for health, query, and citation checks. Expected output:

```text
  ✅  health endpoint reachable
  ✅  /query returns an answer
  ✅  answer contains a CITATIONS section
  ✅  medical disclaimer present
```

### Step 8: (Optional) Launch Gradio demo UI

```bash
python src/gradio_app.py
# Open http://localhost:7860
```

---

## Evaluation

```bash
python eval/run_eval.py --report
# Generates eval/report.html — Evidently AI scores across 5 benchmark queries
```

Metrics: groundedness · relevance · citation rate · disclaimer presence · overall

---

## Troubleshooting

**Connection refused to `localhost:8080`**
→ The model servers aren't running (or were restarted, so the ports in `.env` are stale). Bring them back up and re-wire `.env` with `bash scripts/serve_models.sh` (Step 4).

**FAISS index not found**
→ Run `python scripts/build_index.py` to generate `data/index/merck.faiss` (with the embedding server running).

**Embedding step crashes with `exit code 133`**
→ Known Anaconda Desktop bug on long chunks — keep `CHUNK_SIZE_WORDS` at 200. See **Known issues** below.

---

## Known issues and workarounds

### ⚠️ Anaconda Desktop: Qwen3-Embedding-8B crashes on chunks > 512 tokens (exit code 133)

**Symptom:** `build_index.py` runs successfully for several chunks, then `RemoteDisconnected` error mid-run. Desktop shows "Errored" badge with exit code 133.

**Root cause:** Anaconda Desktop's current llama.cpp build (b8994) crashes with a SIGTRAP (assertion failure) when processing a text chunk that requires multi-pass batching — i.e., any chunk whose token count exceeds the server's `n_batch` limit of 512. This is triggered during LAST-token pooling (the pooling method Qwen3-Embedding uses to produce the final embedding vector) after a split batch. It is a bug in this Desktop build, not in the model itself.

**Current workaround:** `CHUNK_SIZE_WORDS` is set to **200 words** (instead of the ideal 350) to keep all chunks safely under the 512-token limit. Medical text tokenizes at roughly 2 tokens/word, so 200 words ≈ 400 tokens — a safe margin.

**Quality impact:** Smaller chunks mean less context per retrieved passage. To partially compensate, consider increasing `TOP_K` in `.env` from 5 to 8. This retrieves more chunks and preserves total context volume sent to the inference model.

**Proper fixes (pending):**

| Fix | How | Status |
|---|---|---|
| Launch the embedder with a smaller context window via `anaconda ai` | Would reduce KV cache and avoid the multi-pass batching that triggers the crash | `anaconda ai` now works (backend fix below), but `launch` doesn't yet expose a `--ctx-size` flag — so keep `CHUNK_SIZE_WORDS=200` |
| Update Anaconda Desktop | The multi-pass LAST-pooling bug should be fixed in a future Desktop release | Report via Desktop → Support |
| Increase `TOP_K` | `TOP_K=8` in `.env` partially compensates for the smaller chunk size | ✅ Available now |

---

### ✅ `anaconda ai` "401 / API Port not found" on `localhost:8001` — RESOLVED

**Symptom:** `anaconda ai` commands fail with `AINavigatorConfigError: The API Port was not found in the application config file` (or a `401 Unauthorized` from `localhost:8001`).

**Root cause:** Not an auth bug. The CLI was defaulting to the `ai-navigator` backend, which has no configured port — while the backend actually running (as part of Anaconda Desktop) is `anaconda-desktop`, on `localhost:8001`. It was a **backend-selection** mismatch.

**Fix (one time):**
```bash
anaconda ai config --backend anaconda-desktop -y
```

After this, `anaconda ai models`, `launch`, and `servers` all work against the running Desktop backend — which is exactly what makes the two-simultaneous-servers setup in Step 4 (and `scripts/serve_models.sh`) possible. No more manual model-swapping.

---

## High-value packages showcased

| Package | Role | Source |
|---|---|---|
| **Anaconda Desktop** | Local model server — Qwen2.5-14B-Instruct (inference) + Qwen3-Embedding-8B (embeddings). Zero HuggingFace Hub calls; weights served from Anaconda's curated catalog | [Anaconda Desktop](https://www.anaconda.com/products/desktop) |
| **FAISS** | Vector similarity search (CPU) | Anaconda `main` (faiss-cpu) |
| **Gradio** | Interactive demo UI | Anaconda `main` |
| **Evidently AI** | RAG evaluation and monitoring | Anaconda `main` (added Q1 2026) |
| **FastAPI** | Local REST API | Anaconda `main` |
| **requests** | All model API calls (embeddings + chat completions) — no SDKs | Anaconda `main` |

---

## Extension challenges (optional)

Finished the build? Try one of these to make it your own — each is optional and open-ended:

- **Add a topic.** Add a new Merck Manual URL to `scripts/scraper.py`, re-run scrape + `build_index.py`, and ask a question about it. Did retrieval surface your new content?
- **Tune retrieval.** Change `TOP_K` in `.env` (try 3, then 8) and re-run the eval harness. How do groundedness and latency trade off?
- **Swap the model.** Point `INFERENCE_MODEL` at a different chat model in your Anaconda Desktop catalog and compare answer quality on the 5 benchmark queries — no code changes required.
- **Bring your own corpus.** Replace the Merck scraper with a different public dataset and adapt the chunking. The rest of the pipeline is domain-agnostic.

## What's next

- **Streaming responses** — Anaconda Desktop's OpenAI-compatible API supports SSE; wire it through FastAPI + Gradio
- **Multi-turn conversation** — maintain chat history in the Gradio UI
- **More Merck topics** — the scraper supports any URL; expand the topic list
- **RAGAS evaluation** — swap in RAGAS for more rigorous RAG-specific metrics

---

## Learning more

- [Anaconda Desktop](https://www.anaconda.com/products/desktop)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
- [Anaconda CLI](https://anaconda.sh)
- [Merck Manual Professional Edition](https://www.merckmanuals.com/professional)
- [Evidently AI docs](https://docs.evidentlyai.com)
- [MCP + conda + Claude Desktop tutorial](https://github.com/dbouquin/mcp_conda_claude_tutorial) — Daina Bouquin's hands-on intro to building MCP servers with Python and conda (NYT Books API example)

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

If you find a factual error in a generated clinical answer, please open an issue — that's exactly the kind of signal that makes RAG evaluation better.

---

## Acknowledgments

- Built with [FAISS](https://github.com/facebookresearch/faiss), [FastAPI](https://fastapi.tiangolo.com), and [Anaconda Desktop](https://www.anaconda.com/products/desktop)
- Clinical content sourced from the [Merck Manual Professional Edition](https://www.merckmanuals.com/professional) (educational use, robots.txt compliant)
- Demo format inspired by [@dbouquin](https://github.com/dbouquin)'s Anaconda tutorial series

---

> ⚠️ **This project is for educational and demonstration purposes only. Nothing in this repository or its outputs constitutes medical advice. Always consult a qualified healthcare professional.**
