# 🏥 Clinical Knowledge RAG — vLLM + Anaconda CLI + Anaconda Platform AI Orchestration

> **Owner:** Federico Rubiano ([@federicorubiano](https://github.com/federicorubiano)) | **Last tested:** 2026-05-18 | **Status:** Active | **Estimated time:** 60–90 minutes

A production RAG system that answers clinical questions grounded in the **Merck Manual Professional Edition**, powered by vLLM inference, FAISS dense retrieval with cross-encoder reranking, and deployed via Anaconda CLI + Outerbounds.

This demo fills the vLLM gap in Anaconda's high-value AI packages portfolio and tells a complete **"first install to production"** story:
`ana login` → `conda env create` → `ana ob deploy`

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

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. Here it powers semantic retrieval — finding chunks whose *meaning* matches the query, not just their keywords. A cross-encoder then reranks the top candidates for maximum relevance.

## What is Outerbounds?

[Outerbounds](https://outerbounds.com) is a production ML platform built on Metaflow, now part of Anaconda. It provides GPU-backed Kubernetes infrastructure so you can deploy the same code that runs locally straight to production — without rewriting anything. `ana ob deploy` is the single command that takes this project from laptop to live endpoint.

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
| BM25 only | Missed semantic similarity | FAISS dense retrieval → cross-encoder rerank |

---

## Architecture

```
User Query
    │
    ▼
DenseRetriever
    └── FAISS dense search → CrossEncoder rerank → top-4 chunks
    │
    ▼
VLLMClient  (Mistral-7B-Instruct via vLLM)
    │   System: citation rules + structure enforcement
    │   User:   context blocks + question
    ▼
Answer + Citations + Medical Disclaimer  [appended to every response]
    │
    └── FastAPI  /query  →  Gradio UI  →  Browser

On Outerbounds (production):
    [Kubernetes GPU pod]
        ├── vLLM server  (GPU)  → localhost:8001/v1
        └── FastAPI      (CPU)  → 0.0.0.0:8000  ← public endpoint
```

---

## Prerequisites

- Anaconda or Miniconda installed
- [Anaconda CLI (`ana`)](https://anaconda.sh) installed
- Python 3.11
- An Outerbounds account — required for GPU inference (vLLM has no Apple Silicon build; NVIDIA GPU needed for local inference)

> **Mac users:** scraping, index building, and API testing all work locally. vLLM inference requires `ana ob deploy` — there is no local path on Apple Silicon.

---

## Project structure

```
vllm-clinical-rag/
├── src/
│   ├── api.py              # FastAPI /query + /health endpoints
│   ├── retriever.py        # FAISS dense retrieval + cross-encoder reranking
│   ├── vllm_client.py      # vLLM HTTP client + prompt builder (uses requests)
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

## Setup

### Step 1: Login

```bash
ana login
```

`faiss-cpu`, `sentence-transformers`, `gradio`, and `vllm` (CPU) are all on the Anaconda `main` channel. The GPU build of `vllm` is in progress. The environment files handle all package installation — no manual pip commands needed.

### Step 2: Create the environment

**On a machine with an NVIDIA GPU** (or for Outerbounds deployment):
```bash
conda env create -f environment.yml
conda activate vllm-clinical-rag
```

**On a Mac or any CPU-only machine** (scraping, index building, and remote API testing work; vLLM inference runs on Outerbounds):
```bash
conda env create -f environment-local.yml
conda activate vllm-clinical-rag-local
```

### Step 3: Configure your environment

```bash
cp .env.example .env
# Edit .env — set VLLM_MODEL and any overrides
```

### Step 4: Scrape the Merck Manual and build indexes

```bash
python scripts/scraper.py      # ~5 min — respects robots.txt 5s crawl delay
python scripts/build_index.py  # ~10 min — embeds all chunks with gte-large
```

This writes to `data/raw/` and `data/index/`. No corpus file to download — anyone can reproduce it.

### Step 5: Start vLLM (local GPU only)

```bash
bash scripts/start_vllm.sh
# Or with a smaller model:
bash scripts/start_vllm.sh --model mistralai/Mistral-3B-Instruct
```

> **No GPU?** Skip this step and use `ana ob deploy` instead (see Step 8).

### Step 6: Start the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 7: Test your setup

```bash
python test_api.py
```

You should see green checkmarks for health, query, and citation checks.

### Step 8: Deploy to Outerbounds (required for GPU inference)

```bash
ana ob init       # first time only — registers the project
ana ob configure  # paste your platform token from your Outerbounds admin
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
→ Expected — vLLM has no osx-arm64 conda build. Use `environment-local.yml` which omits vllm, then deploy to Outerbounds for GPU inference via `ana ob deploy`.

**`ana ob deploy` says authorization required**
→ Run `ana ob configure <token>` first. Get your token from your Outerbounds admin.

**FAISS index not found**
→ Run `python scripts/build_index.py` to generate `data/index/merck.faiss`.

**vLLM OOM (out of memory)**
→ Try a smaller model: `--model mistralai/Mistral-3B-Instruct` or reduce `--gpu-memory-utilization 0.80`.

---

## High-value packages showcased

| Package | Role | Source |
|---|---|---|
| **vLLM** | Production LLM inference — PagedAttention, concurrent batching | Anaconda `main` (linux-64); GPU build in progress |
| **FAISS** | Vector similarity search — CPU locally, GPU on Outerbounds | Anaconda `main` (faiss-cpu + faiss-gpu) |
| **Gradio** | Interactive demo UI | Anaconda `main` |
| **Evidently AI** | RAG evaluation and monitoring | Anaconda `main` (added Q1 2026) |
| **sentence-transformers** | Text embeddings + cross-encoder reranking | Anaconda `main` |
| **FastAPI** | Production REST API | Anaconda `main` |

---

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
- [Outerbounds documentation](https://docs.outerbounds.com)
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
