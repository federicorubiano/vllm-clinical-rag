#!/usr/bin/env bash
#
# serve_models.sh — launch the two Anaconda Desktop model servers (chat +
# embedder) via the `anaconda ai` CLI and write their (ephemeral) URLs + names
# into .env. This uses the `anaconda ai` server stack — the supported
# way to run TWO models at once (the Desktop UI serves only one at a time).
#
# Defaults to the recommended pair for a 32 GB machine:
#   chat     : Qwen2.5-14B-Instruct/Q4_K_M   (~10 GB, non-thinking)
#   embedder : Qwen3-Embedding-8B/Q8_0        (~12 GB, 4096-dim)
#
# Override by exporting INFER_SPEC / EMBED_SPEC, e.g.:
#   INFER_SPEC=Qwen2.5-7B-Instruct/Q4_K_M EMBED_SPEC=Qwen3-Embedding-0.6B/Q4_K_M bash scripts/serve_models.sh
#
# NOTE: `anaconda ai launch` assigns a RANDOM port per server, so the URLs in
# .env are session-specific. Re-run this script whenever you restart the servers.
# Changing the EMBEDDING model changes the vector dimension — REBUILD the index:
#   python scripts/build_index.py
#
set -uo pipefail
export PATH="$HOME/.ana/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

INFER_SPEC="${INFER_SPEC:-Qwen2.5-14B-Instruct/Q4_K_M}"
EMBED_SPEC="${EMBED_SPEC:-Qwen3-Embedding-8B/Q8_0}"
ENV_FILE=".env"

if ! command -v anaconda >/dev/null 2>&1; then
  echo "ERROR: 'anaconda' CLI not found. Is Anaconda Desktop installed?" >&2
  exit 1
fi

# Launch a model only if a server for it isn't already running (idempotent).
ensure_running() {
  local spec="$1" frag="${1%%/*}"
  if anaconda ai servers --json 2>/dev/null | grep -qF "$frag"; then
    echo "  already running: $frag"
  else
    echo "  launching: $spec (downloads first if needed)"
    if ! anaconda ai launch "$spec" --detach >/dev/null 2>&1; then
      echo "  ERROR: failed to launch $spec — check 'anaconda ai models $frag'" >&2
      return 1
    fi
  fi
}

echo "Ensuring model servers are up..."
ensure_running "$INFER_SPEC"
ensure_running "$EMBED_SPEC"

# Resolve each server's OpenAI-compatible URL and patch .env.
python3 - "$INFER_SPEC" "$EMBED_SPEC" "$ENV_FILE" <<'PY'
import sys, json, subprocess, re, os

infer_spec, embed_spec, env_file = sys.argv[1], sys.argv[2], sys.argv[3]
infer_name, embed_name = infer_spec.split("/")[0], embed_spec.split("/")[0]

def _json(args, default):
    out = subprocess.run(args, capture_output=True, text=True).stdout
    try:
        return json.loads(out or "")
    except Exception:
        return default

def url_for(frag):
    for s in _json(["anaconda", "ai", "servers", "--json"], []):
        if frag in s.get("model", ""):
            det = _json(["anaconda", "ai", "servers", s["server_id"], "--json"], {})
            return det.get("openai_url", "")
    return ""

infer_url, embed_url = url_for(infer_name), url_for(embed_name)
if not infer_url or not embed_url:
    sys.stderr.write(f"ERROR: could not resolve URLs (chat={infer_url!r}, embed={embed_url!r}). "
                     f"Run: anaconda ai servers\n")
    sys.exit(1)

updates = {
    "INFERENCE_MODEL": infer_name,
    "EMBEDDING_MODEL": embed_name,
    "INFERENCE_API_URL": infer_url,
    "EMBEDDING_API_URL": embed_url,
}

lines = open(env_file).read().splitlines() if os.path.exists(env_file) else []
seen = set()
for i, ln in enumerate(lines):
    m = re.match(r"^(\w+)=", ln)
    if m and m.group(1) in updates:
        lines[i] = f"{m.group(1)}={updates[m.group(1)]}"
        seen.add(m.group(1))
for k, v in updates.items():
    if k not in seen:
        lines.append(f"{k}={v}")
open(env_file, "w").write("\n".join(lines) + "\n")

print("Wired .env:")
for k, v in updates.items():
    print(f"  {k}={v}")
PY

echo ""
echo "Servers:"
anaconda ai servers 2>/dev/null
echo ""
echo "Reminder: ports are ephemeral — re-run this script after restarting servers."
echo "If you changed the EMBEDDING model, rebuild the index: python scripts/build_index.py"
