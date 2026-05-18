"""
ob_app.py
---------
Outerbounds App definition for the Clinical Knowledge API.
Deploys a GPU-backed vLLM inference server + FastAPI endpoint as a
persistent service on the Outerbounds platform.

Deploy:
    ana ob init          # first time only — creates project on platform
    ana ob deploy        # deploys this file to Outerbounds

Check:
    ana ob check         # verify packages + config before deploying
    ana ob app list      # list running apps and their endpoints

The deployed service exposes:
    POST /query          — answer a clinical question (same as local)
    GET  /health         — liveness + model info

Once deployed, copy the endpoint URL into your .env:
    API_URL=https://<your-outerbounds-endpoint>

Then point Claude Desktop's MCP server at the same URL via VLLM_BASE_URL.
"""

import subprocess
import time
import os
import signal
import sys

from metaflow import FlowSpec, step, kubernetes, environment, current

# ── Resource configuration ────────────────────────────────────────────────────
# Adjust GPU type and count to match what's available in your Outerbounds
# perimeter. Common options: nvidia-tesla-t4, nvidia-tesla-a10g, nvidia-a100
GPU_TYPE  = os.getenv("OB_GPU_TYPE",  "nvidia-tesla-t4")
GPU_COUNT = int(os.getenv("OB_GPU_COUNT", "1"))
CPU_COUNT = int(os.getenv("OB_CPU_COUNT", "4"))
MEMORY_GB = int(os.getenv("OB_MEMORY_GB", "32"))

VLLM_MODEL   = os.getenv("VLLM_MODEL",   "mistralai/Mistral-7B-Instruct-v0.3")
VLLM_PORT    = int(os.getenv("VLLM_PORT", "8001"))
API_PORT     = int(os.getenv("API_PORT",  "8000"))


class ClinicalRAGApp(FlowSpec):
    """
    Outerbounds App: Clinical Knowledge API
    ----------------------------------------
    Runs vLLM inference + FastAPI RAG server as a persistent GPU service.

    Architecture on Outerbounds:
        [Kubernetes pod]
            ├── vLLM server  (GPU)  → localhost:8001/v1
            └── FastAPI API  (CPU)  → 0.0.0.0:8000  ← public endpoint
    """

    @step
    def start(self):
        """Entry point — validates environment before launching services."""
        print(f"Clinical RAG App starting...")
        print(f"  Model      : {VLLM_MODEL}")
        print(f"  GPU type   : {GPU_TYPE} x{GPU_COUNT}")
        print(f"  vLLM port  : {VLLM_PORT}")
        print(f"  API port   : {API_PORT}")
        self.next(self.serve)

    @kubernetes(
        gpu=GPU_COUNT,
        cpu=CPU_COUNT,
        memory=MEMORY_GB * 1024,   # Metaflow expects MB
        gpu_vendor="nvidia",
        image="anaconda/vllm-clinical-rag:latest",  # built from Dockerfile
    )
    @environment(vars={
        "VLLM_MODEL":      VLLM_MODEL,
        "VLLM_PORT":       str(VLLM_PORT),
        "API_PORT":        str(API_PORT),
        "PYTHONPATH":      "/app",
        "HF_HOME":         "/tmp/hf_cache",
    })
    @step
    def serve(self):
        """
        Launch vLLM + FastAPI on a GPU Kubernetes pod.
        This step runs indefinitely — Outerbounds manages the lifecycle.
        """
        import threading

        # ── Start vLLM in background ──────────────────────────────────────────
        print(f"[serve] Starting vLLM server (model: {VLLM_MODEL})...")
        vllm_proc = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  VLLM_MODEL,
            "--host",                   "0.0.0.0",
            "--port",                   str(VLLM_PORT),
            "--tensor-parallel-size",   str(GPU_COUNT),
            "--gpu-memory-utilization", "0.90",
            "--dtype",                  "float16",
            "--max-model-len",          "4096",
            "--served-model-name",      VLLM_MODEL.split("/")[-1],
        ])

        # ── Wait for vLLM to become ready ─────────────────────────────────────
        import requests
        print(f"[serve] Waiting for vLLM to be ready on port {VLLM_PORT}...")
        for attempt in range(60):                # up to 5 minutes
            try:
                r = requests.get(
                    f"http://localhost:{VLLM_PORT}/health", timeout=5
                )
                if r.status_code == 200:
                    print(f"[serve] vLLM ready after {attempt * 5}s ✓")
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            print("[serve] ERROR: vLLM did not become ready in time.")
            vllm_proc.terminate()
            raise RuntimeError("vLLM startup timeout")

        # ── Start FastAPI in background ───────────────────────────────────────
        print(f"[serve] Starting FastAPI on port {API_PORT}...")
        api_proc = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "src.api:app",
            "--host", "0.0.0.0",
            "--port", str(API_PORT),
            "--workers", "2",
        ])

        print(f"[serve] Clinical Knowledge API is live on port {API_PORT} ✓")
        print(f"[serve] Endpoint: POST /query | GET /health")

        # ── Keep alive — handle graceful shutdown ─────────────────────────────
        def _shutdown(sig, frame):
            print("[serve] Shutdown signal received. Stopping services...")
            api_proc.terminate()
            vllm_proc.terminate()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT,  _shutdown)

        # Health check loop — restarts child processes if they crash
        while True:
            if vllm_proc.poll() is not None:
                print("[serve] WARNING: vLLM process exited — restarting...")
                vllm_proc = subprocess.Popen(vllm_proc.args)
            if api_proc.poll() is not None:
                print("[serve] WARNING: FastAPI process exited — restarting...")
                api_proc = subprocess.Popen(api_proc.args)
            time.sleep(30)

    @step
    def end(self):
        """Teardown step — reached only on explicit stop."""
        print("Clinical RAG App stopped.")


if __name__ == "__main__":
    ClinicalRAGApp()
