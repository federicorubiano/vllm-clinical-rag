# 🏥 Clinical Knowledge RAG — vLLM + Anaconda CLI + Anaconda Platform AI Orchestration

> **Owner:** Federico Rubiano ([@federicorubiano](https://github.com/federicorubiano)) | **Last tested:** 2026-05-18 | **Status:** Active | **Estimated time:** 60–90 minutes

A hands-on **guide**: build your own clinical-question RAG system, step by step, grounded in the **Merck Manual Professional Edition** — entirely on Anaconda's secure-by-default stack. By the end you'll have a working, locally-served RAG you can extend.

Powered by Anaconda Desktop local inference (Qwen3-8B), FAISS dense retrieval with Qwen3-Embedding-4B instruction-following embeddings, and an optional production path via Anaconda CLI + Anaconda Platform AI Orchestration.

> *"If you can't reproduce it, you can't trust it. If you can't trust it, you can't ship it."*

Every model weight enters through Anaconda's curated catalog — auditable provenance from install to inference, zero HuggingFace Hub calls at runtime. This is what **secure by default** looks like end to end:
`ana login` → `conda env create` → open Anaconda Desktop → run locally → *(optional)* `ana ob deploy`

## Audience

Python developers and data scientists who want to build a retrieval-augmented generation (RAG) system end to end using only Anaconda's curated, secure-by-default stack. No prior RAG experience required. The techniques apply to any domain that needs grounded, citable answers over a document corpus — demonstrated here on a clinical/healthcare knowledge base.

## What you'll learn

By the end of this guide you will be able to:

1. **Build** a FAISS dense-vector index from a scraped corpus using Anaconda Desktop's local embedding server — no GPU, no HuggingFace Hub.
2. **Retrieve** the most relevant passages for a question with instruction-following embeddings.
3. **Generate** grounded, citation-enforced answers from a locally-served Qwen3-8B model.
4. **Serve** the pipeline as a FastAPI endpoint.
5. **Evaluate** answer quality with a reproducible scoring harness.

*Optional capstone:* deploy the same code to a GPU endpoint on Anaconda Platform AI Orchestration.

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

## What is vLLM?

[vLLM](https://github.com/vllm-project/vllm) is an open-source library for fast LLM inference and serving. The key innovation is **PagedAttention** — a memory management technique that lets the GPU handle multiple requests concurrently by sharing KV cache memory, similar to how operating systems page virtual memory.

**Why it matters for this demo:** the V1 system (llama-cpp) processed one query at a time. vLLM serves concurrent requests at production scale with the same model. That's the before/after moment this demo makes tangible.

## What is FAISS?

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. Here it powers semantic retrieval — finding chunks whose *meaning* matches the query, not just their keywords. Qwen3-Embedding-4B's instruction-following embeddings handle query/document asymmetry natively, so no separate reranking step is needed.

## What is Anaconda Platform AI Orchestration?

**Anaconda Platform AI Orchestration** — formerly [Outerbounds](https://outerbounds.com), now part of Anaconda — brings production orchestration and observability to the Anaconda platform. Anaconda already owns the trusted foundation in development — secure environments, curated packages, governed models. AI Orchestration completes the path: the same code that runs locally deploys straight to a GPU-backed Kubernetes endpoint, with full lineage, monitoring, and guaranteed SLAs. Together, you get a single auditable path from experimentation to production. `ana ob deploy` is the command that closes that gap.

## What is Evidently AI?

[Evidently AI](https://www.evidentlyai.com) is an open-source library for evaluating and monitoring ML models and LLM pipelines. In this demo it is available as a conda dependency from the Anaconda `main` channel (added Q1 2026) — no pip required. The evaluation harness uses custom heuristic scoring for full reproducibility (no LLM-as-judge); Evidently's richer metrics are available for deeper analysis.

---

## V1 → V2: What changed and why

| Failure mode (V1) | Root cause | V2 fix |
|---|---|---|
| Hallucinated citations | LLM invented sources | Retrieved chunk metadata enforced in response schema |
| Outdated corpus | Watermarked PDF, no update path | Live scraper targeting merckmanuals.com |
| Single-threaded inference | llama-cpp; no concurrency | vLLM with PagedAttention + continuous batching |
| Self-judging evaluation | Mistral scored its own output | Custom heuristic scoring — no LLM-as-judge, fully reproducible; Evidently available for deeper analysis |
| BM25 only | Missed semantic similarity | FAISS dense retrieval with Qwen3-Embedding-4B instruction-following embeddings — no separate reranker needed |
| Dependency on external APIs | Reddit V2 roadmap asked about migrating to hosted models (Claude/OpenAI API) | V2 went the opposite direction: fully self-hosted vLLM — no external API calls, data never leaves your infrastructure |

---

## Architecture

```
User Query
    │
    ▼
DenseRetriever
    └── Qwen3-Embedding-4B (query instruction prefix)
        → FAISS dense search → top-5 chunks
    │   [No cross-encoder — instruction-following asymmetry handles it]
    │
    ▼
DesktopClient  (Qwen3-8B via Anaconda Desktop local server)
    │   System: citation rules + structure enforcement
    │   User:   context blocks + question
    ▼
Answer + Citations + Medical Disclaimer  [appended to every response]
    │
    └── FastAPI  /query  →  Gradio UI  →  Browser

Local development:
    [Anaconda Desktop]
        ├── Qwen3-Embedding-4B server  → localhost:8080/v1/embeddings
        └── Qwen3-8B server            → localhost:8080/v1/chat/completions

On Anaconda Platform (production):
    [Kubernetes GPU pod]
        ├── vLLM  (Qwen3-8B, GPU)  → localhost:8080/v1  ← same API surface
        └── FastAPI      (CPU)     → 0.0.0.0:8000  ← public endpoint
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
- [Anaconda Desktop](https://www.anaconda.com/products/desktop) — provides the local model server for embeddings and inference
- Python 3.11 (installed by the environment file)
- ~24–32 GB RAM recommended to run Qwen3-8B locally
- *(Optional, production only)* an [Anaconda Platform](https://www.anaconda.com/products/platform) account for AI Orchestration

**External dependencies (classified):**
- **Anaconda Desktop model server** — *Tier 3 (no fallback)*: required; serves both the embedding and inference models. This is the point of the secure-by-default story, so the dependency is intentional.
- **Merck Manual website** (scraping step) — *Tier 2 (fallback available)*: a public source; if unreachable, substitute the small sample corpus in `data/raw/`.
- **Anaconda Platform AI Orchestration** — *Tier 3 (no fallback)* but **optional** — only needed for the production-deployment section.

> **Mac users:** the full local pipeline (scraping, indexing, API, and UI) runs on Apple Silicon via Anaconda Desktop. No GPU required for development. Anaconda Platform AI Orchestration is the optional production path.

---

## Project structure

```
vllm-clinical-rag/
├── src/
│   ├── api.py              # FastAPI /query + /health endpoints
│   ├── retriever.py        # FAISS dense retrieval (instruction-following embeddings, no reranker)
│   ├── vllm_client.py      # Anaconda Desktop chat client + prompt builder (uses requests)
│   └── gradio_app.py       # Gradio demo UI
├── scripts/
│   ├── scraper.py          # Merck Manual web scraper (robots.txt compliant)
│   ├── build_index.py      # Builds FAISS index + chunk metadata from scraped text
│   └── start_vllm.sh       # Helper to launch local vLLM server
├── eval/
│   └── run_eval.py         # Heuristic evaluation harness (5 benchmark queries)
├── notebooks/
│   └── demo.ipynb          # End-to-end walkthrough notebook
├── data/
│   ├── raw/                # Scraped .txt files (git-ignored; reproduce via scraper)
│   └── index/              # FAISS index + chunk metadata (git-ignored; reproduce via build_index)
├── ob_app.py               # Outerbounds deployment app (ana ob deploy reads this)
├── Dockerfile              # CUDA base image for Outerbounds
├── environment.yml         # Conda env — GPU/Outerbounds; all packages from Anaconda main
├── environment-local.yml   # Conda env — Mac/CPU dev; vllm omitted (no Apple Silicon build)
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

`faiss-cpu`, `gradio`, `evidently`, `fastapi`, and all supporting packages are on the Anaconda `main` channel. The environment files handle all installation — no pip commands, no HuggingFace packages.

### Step 2: Create the environment

**On a machine with an NVIDIA GPU** (or for Anaconda Platform deployment):
```bash
conda env create -f environment.yml
conda activate vllm-clinical-rag
```

**On a Mac or any CPU-only machine** (full local dev via Anaconda Desktop — scraping, indexing, API, and UI all work):
```bash
conda env create -f environment-local.yml
conda activate vllm-rag
```

### Step 3: Configure your environment

```bash
cp .env.example .env
# Edit .env — set INFERENCE_MODEL / EMBEDDING_MODEL to match your Desktop servers
```

### Step 4: Scrape the Merck Manual and build the index

*Start state: your conda env is activated and `.env` is configured (Steps 2–3).*

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

> ℹ️ The embedding server must be the one running in Anaconda Desktop for this step (see Step 5). If you hit an `exit code 133` crash, see **Known issues** below — keep `CHUNK_SIZE_WORDS` at 200.

### Step 5: Start model servers in Anaconda Desktop

Open Anaconda Desktop and start two model servers:

1. **Embedding model** — find `Qwen3-Embedding-4B` in the model catalog → click **Start Server**
2. **Inference model** — find `Qwen3-8B` → click **Start Server**

Both will be available at `localhost:8080`. The `.env` file controls which model is used for each role — no code changes needed to swap models.

> **This replaces the old vLLM server step.** No GPU required locally. Model weights are downloaded once through Anaconda Desktop's curated catalog — no HuggingFace Hub calls at runtime.

**✅ Checkpoint:** `curl http://localhost:8080/health` returns `{"status":"ok"}` (or the Desktop UI shows the server as **Running**).

> ℹ️ Desktop currently serves one model at a time. Run the **embedding** model while building the index (Step 4), then switch to the **inference** model for Steps 6–9. See **Known issues** for why.

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

### Step 8: Deploy to Anaconda Platform AI Orchestration (required for GPU inference)

```bash
ana ob init       # first time only — registers the project
ana ob configure  # paste your platform token from your Anaconda Platform admin
ana ob check      # verify GPU + packages are compatible
ana ob deploy     # push to production GPU endpoint
```

### Step 9: (Optional) Launch Gradio demo UI

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

**`vllm` not found after `conda env create` on Mac**
→ Expected — vLLM has no osx-arm64 conda build. Use `environment-local.yml` which omits vllm, then deploy to Anaconda Platform for GPU inference via `ana ob deploy`.

**`ana ob deploy` says authorization required**
→ Run `ana ob configure <token>` first. Get your token from your Anaconda Platform admin.

**FAISS index not found**
→ Run `python scripts/build_index.py` to generate `data/index/merck.faiss`.

**vLLM OOM (out of memory)**
→ Try a smaller model: `--model mistralai/Mistral-3B-Instruct` or reduce `--gpu-memory-utilization 0.80`.

---

## Known issues and workarounds

### ⚠️ Anaconda Desktop: Qwen3-Embedding-4B crashes on chunks > 512 tokens (exit code 133)

**Symptom:** `build_index.py` runs successfully for several chunks, then `RemoteDisconnected` error mid-run. Desktop shows "Errored" badge with exit code 133.

**Root cause:** Anaconda Desktop's current llama.cpp build (b8994) crashes with a SIGTRAP (assertion failure) when processing a text chunk that requires multi-pass batching — i.e., any chunk whose token count exceeds the server's `n_batch` limit of 512. This is triggered during LAST-token pooling (the pooling method Qwen3-Embedding uses to produce the final embedding vector) after a split batch. It is a bug in this Desktop build, not in the model itself.

**Current workaround:** `CHUNK_SIZE_WORDS` is set to **200 words** (instead of the ideal 350) to keep all chunks safely under the 512-token limit. Medical text tokenizes at roughly 2 tokens/word, so 200 words ≈ 400 tokens — a safe margin.

**Quality impact:** Smaller chunks mean less context per retrieved passage. To partially compensate, consider increasing `TOP_K` in `.env` from 5 to 8. This retrieves more chunks and preserves total context volume sent to the inference model.

**Proper fixes (pending):**

| Fix | How | Status |
|---|---|---|
| Use `anaconda ai` CLI to pass `--ctx-size 2048` | Launches the server with a smaller context window — reduces KV cache and eliminates multi-pass batching for our chunk sizes | Blocked by `anaconda ai` 401 auth error (see below) |
| Update Anaconda Desktop | The multi-pass LAST-pooling bug should be fixed in a future Desktop release | Report via Desktop → Support |
| Increase `TOP_K` | `TOP_K=8` in `.env` partially compensates for the smaller chunk size | ✅ Available now |

---

### ⚠️ `anaconda ai` plugin: 401 Unauthorized on `localhost:8001`

**Symptom:** `anaconda ai` commands return `HTTPError: 401 Client Error: Unauthorized for url: http://localhost:8001/api/models`.

**Root cause:** The `anaconda ai` plugin authenticates against Desktop's internal backend at port 8001. The auth token is not being passed correctly — likely a keyring or site configuration issue.

**Workaround:** Use Anaconda Desktop UI to manage model servers until resolved.

**Diagnosis steps:**
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
anaconda ai config
anaconda sites list
```

**Why this matters:** Fixing the 401 would allow `anaconda ai launch` to run the embedding and inference models on **separate ports simultaneously** — currently Desktop's UI only supports one active server at a time, requiring a manual swap between embedding and inference model.

---

---

## High-value packages showcased

| Package | Role | Source |
|---|---|---|
| **Anaconda Desktop** | Local model server — Qwen3-8B (inference) + Qwen3-Embedding-4B (embeddings). Zero HuggingFace Hub calls; weights served from Anaconda's curated catalog | [Anaconda Desktop](https://www.anaconda.com/products/desktop) |
| **FAISS** | Vector similarity search — CPU locally, GPU on Anaconda Platform | Anaconda `main` (faiss-cpu + faiss-gpu) |
| **Gradio** | Interactive demo UI | Anaconda `main` |
| **Evidently AI** | RAG evaluation and monitoring | Anaconda `main` (added Q1 2026) |
| **FastAPI** | Production REST API | Anaconda `main` |
| **requests** | All model API calls (embeddings + chat completions) — no SDKs | Anaconda `main` |

---

## Extension challenges (optional)

Finished the build? Try one of these to make it your own — each is optional and open-ended:

- **Add a topic.** Add a new Merck Manual URL to `scripts/scraper.py`, re-run scrape + `build_index.py`, and ask a question about it. Did retrieval surface your new content?
- **Tune retrieval.** Change `TOP_K` in `.env` (try 3, then 8) and re-run the eval harness. How do groundedness and latency trade off?
- **Swap the model.** Point `INFERENCE_MODEL` at a different chat model in your Anaconda Desktop catalog and compare answer quality on the 5 benchmark queries — no code changes required.
- **Bring your own corpus.** Replace the Merck scraper with a different public dataset and adapt the chunking. The rest of the pipeline is domain-agnostic.

## What's next

- **Streaming responses** — vLLM supports SSE; wire through FastAPI + Gradio
- **Multi-turn conversation** — maintain chat history in the Gradio UI
- **More Merck topics** — the scraper supports any URL; expand the topic list
- **RAGAS evaluation** — swap in RAGAS for more rigorous RAG-specific metrics
- **Fine-tuned reranker** — train a domain-specific cross-encoder on medical QA pairs

---

## Learning more

- [vLLM documentation](https://docs.vllm.ai)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
- [Anaconda Platform AI Orchestration docs (Outerbounds)](https://docs.outerbounds.com)
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

- Built with [vLLM](https://github.com/vllm-project/vllm), [FAISS](https://github.com/facebookresearch/faiss), and [Metaflow](https://metaflow.org)
- Clinical content sourced from the [Merck Manual Professional Edition](https://www.merckmanuals.com/professional) (educational use, robots.txt compliant)
- Deployment infrastructure by [Outerbounds](https://outerbounds.com), now part of [Anaconda](https://anaconda.com)
- Demo format inspired by [@dbouquin](https://github.com/dbouquin)'s Anaconda tutorial series

---

> ⚠️ **This project is for educational and demonstration purposes only. Nothing in this repository or its outputs constitutes medical advice. Always consult a qualified healthcare professional.**
