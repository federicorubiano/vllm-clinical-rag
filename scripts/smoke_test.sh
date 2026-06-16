#!/usr/bin/env bash
#
# smoke_test.sh — End-to-end, NON-DESTRUCTIVE smoke test for the
# Anaconda-Desktop Clinical Knowledge RAG demo (Merck Manual).
#
# What it does (read-only / safe):
#   - Checks the conda env + required Python imports (in the intended env)
#   - Pings the Anaconda Desktop model server (localhost:8080)
#   - Confirms .env and the FAISS index/chunks files exist
#   - Starts the FastAPI app in the BACKGROUND, polls /health, then
#     kills it again on exit (trap)
#   - Runs test_api.py and eval/run_eval.py --report
#   - Judges the eval on the actual scores in eval/results.json (NOT on the
#     Evidently log line — the report renders even over all-zero data)
#   - Reports versions
#
# It NEVER deletes anything, NEVER calls pip, and NEVER touches HuggingFace.
# build_index.py is only run if you explicitly pass --build (and only then
# with the embedding server up). The only file this script removes is a
# mktemp log it created itself.
#
# Usage:
#   bash scripts/smoke_test.sh            # normal run
#   bash scripts/smoke_test.sh --build    # also (re)build the index if missing
#
# When done, copy everything under the final
#   "===== COPY EVERYTHING BELOW BACK TO CLAUDE ====="
# banner and paste it back.

set -uo pipefail

# ---------------------------------------------------------------------------
# Move to repo root. NOTE: the repo path contains a SPACE, so quote always.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.." || {
  echo "FATAL: could not cd to repo root from $(dirname "$0")/.." >&2
  exit 1
}
REPO_ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# Grounded constants (from recon — do not invent).
# ---------------------------------------------------------------------------
EXPECTED_CONDA_ENV="anaconda-clinical-rag"
DESKTOP_API_URL="${DESKTOP_API_URL:-http://localhost:8080/v1}"   # Anaconda Desktop server
API_HOST="0.0.0.0"
API_PORT="8000"
API_URL="http://localhost:${API_PORT}"                            # FastAPI backend
FAISS_PATH="data/index/merck.faiss"
CHUNKS_PATH="data/index/chunks.json"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
SCREENSHOT_PATH="screenshots/gradio-ui.png"
EVAL_RESULTS="eval/results.json"
EVAL_REPORT="eval/report.html"
N_BENCHMARK_QUERIES=5     # eval/run_eval.py BENCHMARK_QUERIES has exactly 5 entries

# Evidently log strings to disambiguate success vs fallback (exact from recon).
# NOTE: these only tell us WHICH report backend ran, NOT whether the RAG
# pipeline actually worked — run_report() emits "Evidently report saved" even
# when every query returned no_response and every score is 0.0. The real
# pass/fail verdict for the pipeline is derived from eval/results.json below.
EVIDENTLY_OK_STR="Evidently report saved"
EVIDENTLY_FALLBACK_STR="Evidently unavailable"

# CLI flag
DO_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) DO_BUILD=1 ;;
    *) echo "WARN: unknown argument '$arg' (ignored)" ;;
  esac
done

# ---------------------------------------------------------------------------
# Summary accounting + check() helper (bash 3.2 friendly — no assoc arrays).
#
# SUMMARY_LINES stores one TAB-separated "STATUS\tLABEL\tDETAIL" record per
# check. INVARIANT: no field may contain a literal TAB (none currently do),
# because the final re-print loop splits on TAB via IFS. Newlines inside a
# detail would also break the table, so keep details single-line.
# ---------------------------------------------------------------------------
SUMMARY_LINES=""
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
WARN_COUNT=0

# check STATUS "label" "optional detail"
#   STATUS in: PASS | FAIL | SKIP | WARN
check() {
  local status="$1"
  local label="$2"
  local detail="${3:-}"
  local mark
  case "$status" in
    PASS) mark="[PASS]"; PASS_COUNT=$((PASS_COUNT + 1)) ;;
    FAIL) mark="[FAIL]"; FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    SKIP) mark="[SKIP]"; SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    WARN) mark="[WARN]"; WARN_COUNT=$((WARN_COUNT + 1)) ;;
    *)    mark="[????]" ;;
  esac
  if [ -n "$detail" ]; then
    printf '  %s  %s — %s\n' "$mark" "$label" "$detail"
  else
    printf '  %s  %s\n' "$mark" "$label"
  fi
  # Record for the final summary (tab-separated; printf keeps it literal).
  SUMMARY_LINES="${SUMMARY_LINES}$(printf '%s\t%s\t%s' "$status" "$label" "$detail")
"
}

header() {
  printf '\n========================================================\n'
  printf '  %s\n' "$1"
  printf '========================================================\n'
}

# Captured error tails for the final paste block.
ERR_TAILS=""
append_err_tail() {
  # append_err_tail "title" "text"
  ERR_TAILS="${ERR_TAILS}
----- $1 -----
$2
"
}

# ---------------------------------------------------------------------------
# Background-server bookkeeping + trap to ALWAYS clean up the uvicorn process.
# ---------------------------------------------------------------------------
UVICORN_PID=""
UVICORN_LOG=""
cleanup() {
  local i=0
  if [ -n "${UVICORN_PID}" ] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo ""
    echo "Cleaning up: stopping FastAPI (uvicorn pid ${UVICORN_PID})..."
    kill "${UVICORN_PID}" 2>/dev/null
    # Give it a moment, then force if still alive (portable poll loop; no `timeout`).
    while [ "$i" -lt 10 ] && kill -0 "${UVICORN_PID}" 2>/dev/null; do
      i=$((i + 1))
      sleep 1
    done
    if kill -0 "${UVICORN_PID}" 2>/dev/null; then
      kill -9 "${UVICORN_PID}" 2>/dev/null
    fi
  fi
  if [ -n "${UVICORN_LOG}" ] && [ -f "${UVICORN_LOG}" ]; then
    rm -f "${UVICORN_LOG}" 2>/dev/null
  fi
}
# EXIT always cleans up. INT/TERM clean up THEN re-raise with the conventional
# exit status (130 / 143) so a wrapping CI/script sees the signal, not a fake 0.
trap 'cleanup' EXIT
trap 'cleanup; trap - INT EXIT; exit 130' INT
trap 'cleanup; trap - TERM EXIT; exit 143' TERM

# ---------------------------------------------------------------------------
# Pick a python interpreter. Prefer the intended conda env's interpreter so we
# test the environment the project actually targets, not whatever python3
# happens to be first on PATH (which may be base and lack the deps entirely).
# Resolution order:
#   1. $CONDA_PREFIX/bin/python  when the active env IS anaconda-clinical-rag
#   2. `conda run -n anaconda-clinical-rag python`  if that env exists
#   3. python3 / python from PATH (last resort; will likely fail imports)
# ---------------------------------------------------------------------------
PY="python"
if command -v python3 >/dev/null 2>&1; then
  PY="python3"
fi

PY_SOURCE="PATH ${PY}"
if [ "${CONDA_DEFAULT_ENV:-}" = "${EXPECTED_CONDA_ENV}" ] \
   && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PY="${CONDA_PREFIX}/bin/python"
  PY_SOURCE="active env (${CONDA_PREFIX})"
elif command -v conda >/dev/null 2>&1 \
     && conda run -n "${EXPECTED_CONDA_ENV}" python -c "pass" >/dev/null 2>&1; then
  # Use an array-free, space-safe wrapper. `conda run` execs python in the env.
  PY="conda-run-py"
  PY_SOURCE="conda run -n ${EXPECTED_CONDA_ENV}"
fi

# run_py: indirection so we can route through `conda run` without arrays.
run_py() {
  if [ "${PY}" = "conda-run-py" ]; then
    conda run -n "${EXPECTED_CONDA_ENV}" python "$@"
  else
    "${PY}" "$@"
  fi
}

# A printable label for the interpreter (for logs / version block).
PY_LABEL="${PY}"
[ "${PY}" = "conda-run-py" ] && PY_LABEL="conda run -n ${EXPECTED_CONDA_ENV} python"

echo "Anaconda Desktop Clinical RAG — smoke test"
echo "Repo root : ${REPO_ROOT}"
echo "Python    : ${PY_LABEL}  (${PY_SOURCE})"
echo "Build flag: $([ "$DO_BUILD" -eq 1 ] && echo on || echo off)"

# ===========================================================================
# (a) Environment: conda env name + python imports
#     HARD PRECONDITION: if neither the expected env is active nor resolvable
#     AND the required imports fail, nothing downstream can pass, so we FAIL
#     and abort (the trap still runs; nothing destructive happens).
# ===========================================================================
header "(a) Environment & Python imports"

ENV_OK=0
if [ "${CONDA_DEFAULT_ENV:-}" = "${EXPECTED_CONDA_ENV}" ]; then
  ENV_OK=1
  check PASS "Conda env active" "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
elif [ "${PY}" = "conda-run-py" ]; then
  ENV_OK=1
  check WARN "Conda env" "'${EXPECTED_CONDA_ENV}' not ACTIVE (CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<unset>}'), but it exists — using 'conda run -n ${EXPECTED_CONDA_ENV} python'. Prefer: conda activate ${EXPECTED_CONDA_ENV}"
else
  check WARN "Conda env" "expected '${EXPECTED_CONDA_ENV}', got '${CONDA_DEFAULT_ENV:-<unset>}', and that env was not found via 'conda run' — run: conda activate ${EXPECTED_CONDA_ENV}"
fi

# Verify imports; report EXACTLY which one fails. pandas is treated as OPTIONAL
# here because it is NOT declared in environment.yml / anaconda-project.yml — it
# arrives only transitively via evidently. A pandas-only miss is a WARN, not a
# FAIL, so a transitive-dep shift does not overstate breakage. The hard-required
# set is faiss, fastapi, gradio, evidently, requests.
IMPORT_OUT="$(run_py - <<'PYEOF' 2>&1
required = ["faiss", "fastapi", "gradio", "evidently", "requests"]
optional = ["pandas"]  # transitive via evidently, not declared in environment.yml
req_failed = []
opt_failed = []
for m in required:
    try:
        __import__(m)
    except Exception as e:  # noqa
        req_failed.append("%s (%s)" % (m, type(e).__name__))
for m in optional:
    try:
        __import__(m)
    except Exception as e:  # noqa
        opt_failed.append("%s (%s)" % (m, type(e).__name__))
if req_failed:
    print("IMPORT_FAIL:" + "; ".join(req_failed))
elif opt_failed:
    print("IMPORT_WARN:" + "; ".join(opt_failed))
else:
    print("IMPORT_OK")
PYEOF
)"
IMPORT_RC=$?
IMPORTS_OK=0
if [ $IMPORT_RC -eq 0 ] && printf '%s' "$IMPORT_OUT" | grep -q "IMPORT_OK"; then
  IMPORTS_OK=1
  check PASS "Python imports" "faiss, fastapi, gradio, evidently, requests (+pandas) all import"
elif [ $IMPORT_RC -eq 0 ] && printf '%s' "$IMPORT_OUT" | grep -q "IMPORT_WARN"; then
  IMPORTS_OK=1
  detail="$(printf '%s' "$IMPORT_OUT" | grep "IMPORT_WARN" | sed 's/^IMPORT_WARN://')"
  check PASS "Python imports (required)" "faiss, fastapi, gradio, evidently, requests OK"
  check WARN "Optional import" "${detail} — pandas is a transitive dep (via evidently), not declared in environment.yml"
else
  detail="$(printf '%s' "$IMPORT_OUT" | grep "IMPORT_FAIL" | sed 's/^IMPORT_FAIL://')"
  [ -z "$detail" ] && detail="$(printf '%s' "$IMPORT_OUT" | tr '\n' ' ')"
  check FAIL "Python imports" "$detail"
  append_err_tail "Python imports" "$IMPORT_OUT"
fi

# If the intended env is neither active nor resolvable AND the required imports
# fail, every downstream step is doomed (uvicorn won't even start). Abort with a
# clear instruction rather than emitting a wall of misleading FAILs.
if [ "$IMPORTS_OK" -ne 1 ] && [ "$ENV_OK" -ne 1 ]; then
  check FAIL "Precondition" "the '${EXPECTED_CONDA_ENV}' env is not active/resolvable and required imports failed — nothing downstream can pass. Run: conda activate ${EXPECTED_CONDA_ENV}  (then re-run this script)"
  header "ABORTED — environment not ready"
  echo ""
  echo "===== COPY EVERYTHING BELOW BACK TO CLAUDE ====="
  echo ""
  echo "Anaconda Desktop Clinical RAG — smoke test ABORTED at step (a)"
  echo "Repo root : ${REPO_ROOT}"
  echo "Date      : $(date 2>/dev/null)"
  echo "Python    : ${PY_LABEL}  (${PY_SOURCE})"
  echo ""
  echo "REASON: conda env '${EXPECTED_CONDA_ENV}' is not active and could not be"
  echo "        resolved via 'conda run', and the required Python imports failed."
  echo "        Fix: conda env create -f environment.yml   (if the env is missing)"
  echo "             conda activate ${EXPECTED_CONDA_ENV}"
  echo "        then re-run: bash scripts/smoke_test.sh"
  echo ""
  echo "--- Import probe output ---"
  printf '%s\n' "$IMPORT_OUT"
  echo ""
  echo "--- Result tally ---"
  echo "PASS=${PASS_COUNT}  FAIL=${FAIL_COUNT}  SKIP=${SKIP_COUNT}  WARN=${WARN_COUNT}"
  echo ""
  echo "===== END — PASTE EVERYTHING ABOVE BACK TO CLAUDE ====="
  exit 1
fi

# ===========================================================================
# (b) Anaconda Desktop model server reachable (informational, but GATES f/g/build)
# ===========================================================================
header "(b) Anaconda Desktop model server (localhost:8080)"

DESKTOP_REACHABLE=0
if command -v curl >/dev/null 2>&1; then
  # Hit the OpenAI-compatible /models endpoint under the base URL.
  # -f makes curl fail (rc!=0) on HTTP >= 400, so a 401/404/500 from a
  # half-up server does NOT count as reachable. This matters because this
  # flag gates the --build path (step d): build_index.py's wait_for_server()
  # only treats a 200 from /models as ready, so we match that precondition.
  DESKTOP_PROBE_URL="${DESKTOP_API_URL%/}/models"
  if curl -fs -o /dev/null --max-time 5 "${DESKTOP_PROBE_URL}"; then
    DESKTOP_REACHABLE=1
    check PASS "Desktop server reachable" "200 from ${DESKTOP_PROBE_URL} (does NOT confirm Qwen3-8B + Qwen3-Embedding-4B are both loaded)"
  else
    check WARN "Desktop server NOT reachable" "no 2xx from ${DESKTOP_PROBE_URL} — start Qwen3-8B AND Qwen3-Embedding-4B servers in Anaconda Desktop (demo can't generate/embed without them)"
  fi
else
  check WARN "Desktop server" "curl not found; skipped reachability probe of ${DESKTOP_API_URL}"
fi

# ===========================================================================
# (c) .env present
# ===========================================================================
header "(c) .env configuration"

if [ -f "${ENV_FILE}" ]; then
  check PASS ".env present" "${REPO_ROOT}/${ENV_FILE}"
else
  if [ -f "${ENV_EXAMPLE}" ]; then
    check WARN ".env missing" "create it with: cp ${ENV_EXAMPLE} ${ENV_FILE}"
  else
    check WARN ".env missing" "and ${ENV_EXAMPLE} not found either"
  fi
fi

# ===========================================================================
# (d) Index present (FAISS + chunks). Optionally build with --build.
# ===========================================================================
header "(d) FAISS index & chunk metadata"

INDEX_OK=0
if [ -f "${FAISS_PATH}" ] && [ -f "${CHUNKS_PATH}" ]; then
  INDEX_OK=1
  if [ "$DO_BUILD" -eq 1 ]; then
    check PASS "Index present" "${FAISS_PATH} + ${CHUNKS_PATH} (already on disk; --build is a no-op, embedding server NOT contacted)"
  else
    check PASS "Index present" "${FAISS_PATH} + ${CHUNKS_PATH}"
  fi
else
  missing=""
  [ -f "${FAISS_PATH}" ]  || missing="${missing} ${FAISS_PATH}"
  [ -f "${CHUNKS_PATH}" ] || missing="${missing} ${CHUNKS_PATH}"
  if [ "$DO_BUILD" -eq 1 ]; then
    if [ "$DESKTOP_REACHABLE" -eq 1 ]; then
      echo "  --build set and Desktop server reachable; running scripts/build_index.py ..."
      BUILD_LOG="$(run_py scripts/build_index.py 2>&1)"
      BUILD_RC=$?
      if [ $BUILD_RC -eq 0 ] && [ -f "${FAISS_PATH}" ] && [ -f "${CHUNKS_PATH}" ]; then
        INDEX_OK=1
        check PASS "Index built" "via scripts/build_index.py (rc=0)"
      else
        check FAIL "Index build failed" "rc=${BUILD_RC}; see error tail. Missing:${missing}"
        append_err_tail "build_index.py (tail)" "$(printf '%s\n' "$BUILD_LOG" | tail -n 25)"
      fi
    else
      check SKIP "Index build" "--build set but Desktop embedding server NOT reachable; build_index.py HARD-REQUIRES a live Qwen3-Embedding-4B server (wait_for_server polls /models for ~60s then raises). Missing:${missing}"
    fi
  else
    check SKIP "Index missing" "missing:${missing}. Run scraper then build WITH the embedding server up: ${PY_LABEL} scripts/scraper.py && ${PY_LABEL} scripts/build_index.py  (or re-run this script with --build). Downstream API/eval steps will be skipped."
  fi
fi

# ===========================================================================
# (e) Start FastAPI in background, poll /health
#     IMPORTANT: a 200 from /health only proves the FastAPI process booted and
#     loaded the FAISS index from disk. It does NOT prove the Desktop inference/
#     embedding server is up — /health builds the retriever + DesktopClient
#     offline (no network call). So we label this precisely and also inspect
#     chunks_loaded in the body.
# ===========================================================================
header "(e) FastAPI app (uvicorn src.api:app :${API_PORT})"

API_READY=0
CHUNKS_LOADED="?"
if [ "$INDEX_OK" -ne 1 ]; then
  check SKIP "FastAPI start" "skipped because the index is not present (see step d)"
else
  # Positional mktemp template — honored by BOTH BSD (macOS) and GNU mktemp,
  # unlike `mktemp -t PREFIX` which on BSD leaves the literal XXXXXX in place.
  UVICORN_LOG="$(mktemp "${TMPDIR:-/tmp}/rag_uvicorn.XXXXXX" 2>/dev/null)" \
    || UVICORN_LOG="/tmp/rag_uvicorn.$$.log"
  echo "  Launching: uvicorn src.api:app --host ${API_HOST} --port ${API_PORT}"
  echo "  (log -> ${UVICORN_LOG})"
  # Start in background; capture stdout+stderr to the temp log.
  if [ "${PY}" = "conda-run-py" ]; then
    conda run -n "${EXPECTED_CONDA_ENV}" python -m uvicorn src.api:app \
      --host "${API_HOST}" --port "${API_PORT}" > "${UVICORN_LOG}" 2>&1 &
  else
    "${PY}" -m uvicorn src.api:app \
      --host "${API_HOST}" --port "${API_PORT}" > "${UVICORN_LOG}" 2>&1 &
  fi
  UVICORN_PID=$!

  # Portable poll loop (~30s max). Do NOT rely on the `timeout` command.
  # -f so a 5xx from /health does NOT count as ready.
  if command -v curl >/dev/null 2>&1; then
    i=0
    while [ "$i" -lt 30 ]; do
      # Bail early if the process already died.
      if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
        break
      fi
      if curl -fs -o /dev/null --max-time 3 "${API_URL}/health"; then
        API_READY=1
        break
      fi
      i=$((i + 1))
      sleep 1
    done
  else
    check WARN "FastAPI health" "curl not found; cannot poll ${API_URL}/health"
  fi

  if [ "$API_READY" -eq 1 ]; then
    HEALTH_JSON="$(curl -fs --max-time 5 "${API_URL}/health" 2>/dev/null)"
    # Pull chunks_loaded out of the JSON body without assuming jq.
    CHUNKS_LOADED="$(printf '%s' "$HEALTH_JSON" \
      | run_py -c 'import sys,json
try:
    print(json.load(sys.stdin).get("chunks_loaded","?"))
except Exception:
    print("?")' 2>/dev/null)"
    [ -z "$CHUNKS_LOADED" ] && CHUNKS_LOADED="?"
    if [ "$CHUNKS_LOADED" = "0" ]; then
      check WARN "FastAPI booted, 0 chunks" "/health 200 but chunks_loaded=0 — index loaded empty; /query will find no context (NOT proof of inference)"
    else
      check PASS "FastAPI booted + index loaded" "/health 200, chunks_loaded=${CHUNKS_LOADED} (process up + FAISS loaded; NOT proof the Desktop model server answers)"
    fi
  else
    if kill -0 "${UVICORN_PID}" 2>/dev/null; then
      check FAIL "FastAPI /health" "no 2xx from ${API_URL}/health within ~30s (server still running)"
    else
      check FAIL "FastAPI start" "uvicorn process exited before becoming ready (likely missing deps in '${PY_LABEL}', or port ${API_PORT} in use)"
    fi
    append_err_tail "uvicorn (tail)" "$(tail -n 25 "${UVICORN_LOG}" 2>/dev/null)"
  fi
fi

# ===========================================================================
# (f) test_api.py
#     Gated on BOTH API_READY and DESKTOP_REACHABLE: test_api.py's [2] Query
#     test hits /query, which calls the live Desktop LLM. With Desktop down,
#     /query 502s and FAILs look like API bugs rather than "Desktop not running".
# ===========================================================================
header "(f) test_api.py connectivity test"

if [ "$API_READY" -ne 1 ]; then
  check SKIP "test_api.py" "skipped because FastAPI is not ready (see step e)"
elif [ "$DESKTOP_REACHABLE" -ne 1 ]; then
  check SKIP "test_api.py" "FastAPI is up but the Anaconda Desktop inference server is NOT reachable (step b) — /query cannot succeed. Start Qwen3-8B + Qwen3-Embedding-4B in Anaconda Desktop, then re-run."
else
  TEST_OUT="$(API_URL="${API_URL}" run_py test_api.py --url "${API_URL}" 2>&1)"
  TEST_RC=$?
  TEST_TAIL="$(printf '%s\n' "$TEST_OUT" | tail -n 20)"
  if [ $TEST_RC -eq 0 ]; then
    check PASS "test_api.py" "rc=0 (all connectivity tests passed)"
  else
    check FAIL "test_api.py" "rc=${TEST_RC} — see error tail (if /query failed, confirm the Desktop model server is actually serving Qwen3-8B)"
    append_err_tail "test_api.py (tail)" "$TEST_TAIL"
  fi
  echo "  --- test_api.py output (tail) ---"
  printf '%s\n' "$TEST_TAIL" | sed 's/^/  /'
fi

# ===========================================================================
# (g) eval/run_eval.py --report  + REAL pipeline verdict from results.json
#     Gated on BOTH API_READY and DESKTOP_REACHABLE for the same reason as (f):
#     run_eval's /query calls hit the live Desktop LLM/embedding server.
#
#     CRITICAL: do NOT treat "Evidently report saved" as a pipeline-health PASS.
#     run_report() emits that string even when all 5 queries returned no_response
#     and every score is 0.0. We therefore judge the pipeline on eval/results.json:
#       - aggregate.queries_scored == 5  AND  aggregate.overall > 0  -> PASS
#       - any per-query error == "no_response"                       -> FAIL
#     and keep the Evidently grep only as a "which report backend ran" note.
# ===========================================================================
header "(g) eval/run_eval.py --report (pipeline verdict + Evidently backend)"

EVIDENTLY_VERDICT="not run"
PIPELINE_VERDICT="not run"
if [ "$API_READY" -ne 1 ]; then
  check SKIP "run_eval.py" "skipped because FastAPI is not ready (see step e)"
  EVIDENTLY_VERDICT="skipped (API not ready)"
  PIPELINE_VERDICT="skipped (API not ready)"
elif [ "$DESKTOP_REACHABLE" -ne 1 ]; then
  check SKIP "run_eval.py" "FastAPI is up but the Anaconda Desktop inference/embedding server is NOT reachable (step b) — every /query would return no_response and the report would render over all-zero data. Start Qwen3-8B + Qwen3-Embedding-4B, then re-run."
  EVIDENTLY_VERDICT="skipped (Desktop not reachable)"
  PIPELINE_VERDICT="skipped (Desktop not reachable)"
else
  # Record the prior results.json mtime so we can confirm THIS run wrote it
  # (avoids judging a stale file if run_eval exits before writing).
  PREV_RESULTS_MTIME=""
  if [ -f "${EVAL_RESULTS}" ]; then
    PREV_RESULTS_MTIME="$(ls -l "${EVAL_RESULTS}" 2>/dev/null)"
  fi

  EVAL_OUT="$(API_URL="${API_URL}" run_py eval/run_eval.py --api-url "${API_URL}" --report 2>&1)"
  EVAL_RC=$?
  EVAL_TAIL="$(printf '%s\n' "$EVAL_OUT" | tail -n 25)"

  if [ $EVAL_RC -eq 0 ]; then
    check PASS "run_eval.py ran" "rc=0 (harness completed; quality judged separately below)"
  else
    check FAIL "run_eval.py" "rc=${EVAL_RC} — see error tail"
    append_err_tail "run_eval.py (tail)" "$EVAL_TAIL"
  fi

  # ---- REAL pipeline verdict from results.json ---------------------------
  if [ -f "${EVAL_RESULTS}" ]; then
    NEW_RESULTS_MTIME="$(ls -l "${EVAL_RESULTS}" 2>/dev/null)"
    if [ "${NEW_RESULTS_MTIME}" = "${PREV_RESULTS_MTIME}" ]; then
      check WARN "results.json (stale)" "${EVAL_RESULTS} exists but was NOT rewritten by this run — run_eval likely exited before writing. Judging skipped."
      PIPELINE_VERDICT="UNKNOWN — results.json not rewritten this run"
    else
      check PASS "results.json present" "${EVAL_RESULTS} (written this run)"
      # Parse aggregate + per-query errors. Emit a single VERDICT line.
      VERDICT_LINE="$(run_py - "${EVAL_RESULTS}" "${N_BENCHMARK_QUERIES}" <<'PYEOF' 2>&1
import json, sys
path = sys.argv[1]; expected = int(sys.argv[2])
try:
    with open(path) as f:
        data = json.load(f)
except Exception as e:
    print("VERDICT\tFAIL\tcould not read %s (%s)" % (path, type(e).__name__)); sys.exit(0)
agg = data.get("aggregate", {}) or {}
results = data.get("results", []) or []
scored = agg.get("queries_scored", 0)
overall = agg.get("overall", 0) or 0
no_resp = [r.get("id", "?") for r in results if r.get("error") == "no_response"]
if no_resp:
    print("VERDICT\tFAIL\t%d/%d queries returned no_response (Desktop model server not answering): %s"
          % (len(no_resp), len(results), ", ".join(no_resp)))
elif scored == expected and overall > 0:
    print("VERDICT\tPASS\tall %d/%d queries scored, aggregate overall=%.3f" % (scored, expected, overall))
elif scored == expected and overall <= 0:
    print("VERDICT\tFAIL\tall %d queries scored but aggregate overall=%.3f (answers empty/zero-scored)" % (expected, overall))
else:
    print("VERDICT\tFAIL\tonly %d/%d queries scored (overall=%.3f)" % (scored, expected, overall))
PYEOF
)"
      V_STATUS="$(printf '%s' "$VERDICT_LINE" | sed -n 's/^VERDICT'$'\t''\([A-Z]*\)'$'\t''.*$/\1/p')"
      V_DETAIL="$(printf '%s' "$VERDICT_LINE" | sed -n 's/^VERDICT'$'\t''[A-Z]*'$'\t''\(.*\)$/\1/p')"
      if [ "$V_STATUS" = "PASS" ]; then
        PIPELINE_VERDICT="PASS — ${V_DETAIL}"
        check PASS "RAG pipeline quality" "${V_DETAIL}"
      else
        [ -z "$V_DETAIL" ] && V_DETAIL="$(printf '%s' "$VERDICT_LINE" | tr '\t\n' '  ')"
        PIPELINE_VERDICT="FAIL — ${V_DETAIL}"
        check FAIL "RAG pipeline quality" "${V_DETAIL}"
      fi
    fi
  else
    check FAIL "results.json missing" "expected ${EVAL_RESULTS} — run_eval did not produce it"
    PIPELINE_VERDICT="FAIL — results.json missing"
  fi

  # report.html artifact check.
  if [ -f "${EVAL_REPORT}" ]; then
    check PASS "report.html present" "${EVAL_REPORT}"
  else
    check FAIL "report.html missing" "expected ${EVAL_REPORT}"
  fi

  # ---- Evidently backend note (which report renderer ran), NOT a verdict --
  if printf '%s' "$EVAL_OUT" | grep -qF "${EVIDENTLY_OK_STR}"; then
    EVIDENTLY_VERDICT="Evidently backend used ('${EVIDENTLY_OK_STR}' fired) — note: this only means the report RENDERED, not that answers were good"
    check PASS "Evidently backend" "real Evidently report rendered (quality is judged by 'RAG pipeline quality' above, not by this line)"
  elif printf '%s' "$EVAL_OUT" | grep -qF "${EVIDENTLY_FALLBACK_STR}"; then
    EVIDENTLY_VERDICT="FALLBACK ('${EVIDENTLY_FALLBACK_STR}' fired) — basic pandas HTML table; Evidently itself NOT used"
    check WARN "Evidently backend" "${EVIDENTLY_VERDICT}"
  else
    EVIDENTLY_VERDICT="UNKNOWN — neither '${EVIDENTLY_OK_STR}' nor '${EVIDENTLY_FALLBACK_STR}' found (run_eval may have exited before run_report)"
    check WARN "Evidently backend" "${EVIDENTLY_VERDICT}"
  fi

  echo "  --- run_eval.py output (tail) ---"
  printf '%s\n' "$EVAL_TAIL" | sed 's/^/  /'
fi

# ===========================================================================
# (h) Gradio screenshot present?
# ===========================================================================
header "(h) Gradio UI screenshot"

if [ -f "${SCREENSHOT_PATH}" ]; then
  check PASS "Screenshot present" "${SCREENSHOT_PATH}"
else
  check WARN "Screenshot missing" "MANUAL step: in the activated env run '${PY_LABEL} src/gradio_app.py', open http://localhost:7860, ask a benchmark query, then save the screenshot to ${SCREENSHOT_PATH}"
fi

# ===========================================================================
# (i) Versions (via python, NOT pip)
# ===========================================================================
header "(i) Versions"

CONDA_VER="$(conda --version 2>/dev/null || echo 'conda: not found')"
PY_VER="$(run_py --version 2>&1)"
echo "  ${CONDA_VER}"
echo "  ${PY_VER}  [${PY_LABEL}]"

PKG_VERS="$(run_py - <<'PYEOF' 2>&1
import importlib
# (display label, import name) — handles the faiss-cpu -> faiss name mismatch.
pkgs = [
    ("faiss-cpu", "faiss"),
    ("evidently", "evidently"),
    ("gradio",    "gradio"),
    ("fastapi",   "fastapi"),
    ("pandas",    "pandas"),   # transitive via evidently; fine to report here
]
for label, mod in pkgs:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "unknown")
        print("  %-12s %s" % (label + ":", v))
    except Exception as e:  # noqa
        print("  %-12s import error (%s)" % (label + ":", type(e).__name__))
PYEOF
)"
echo "$PKG_VERS"

# ===========================================================================
# FINAL PASTE BLOCK
# ===========================================================================
printf '\n\n'
echo "===== COPY EVERYTHING BELOW BACK TO CLAUDE ====="
echo ""
echo "Anaconda Desktop Clinical RAG — smoke test summary"
echo "Repo root : ${REPO_ROOT}"
echo "Date      : $(date 2>/dev/null)"
echo "Python    : ${PY_VER}  [${PY_LABEL} via ${PY_SOURCE}]"
echo "Conda     : ${CONDA_VER}"
echo "Desktop   : $([ "$DESKTOP_REACHABLE" -eq 1 ] && echo "reachable (${DESKTOP_API_URL%/}/models)" || echo "NOT reachable — Qwen3-8B / Qwen3-Embedding-4B likely not started")"
echo ""
echo "--- Result tally ---"
echo "PASS=${PASS_COUNT}  FAIL=${FAIL_COUNT}  SKIP=${SKIP_COUNT}  WARN=${WARN_COUNT}"
echo ""
echo "--- Check-by-check ---"
# Re-print the recorded summary lines as a tidy table. Split on TAB (the record
# delimiter); skip the phantom empty record from the trailing newline.
printf '%s' "$SUMMARY_LINES" | while IFS="$(printf '\t')" read -r st lbl det; do
  [ -z "$st" ] && continue
  if [ -n "$det" ]; then
    printf '  [%-4s] %s — %s\n' "$st" "$lbl" "$det"
  else
    printf '  [%-4s] %s\n' "$st" "$lbl"
  fi
done
echo ""
echo "--- RAG pipeline verdict (from eval/results.json, NOT the Evidently log line) ---"
echo "  ${PIPELINE_VERDICT}"
echo ""
echo "--- Evidently report backend ---"
echo "  ${EVIDENTLY_VERDICT}"
echo ""
echo "--- Key versions ---"
echo "  ${CONDA_VER}"
echo "  ${PY_VER}"
printf '%s\n' "$PKG_VERS"
echo ""
echo "--- Captured error tails ---"
if [ -n "$(printf '%s' "$ERR_TAILS" | tr -d '[:space:]')" ]; then
  printf '%s\n' "$ERR_TAILS"
else
  echo "  (none)"
fi
echo ""
echo "===== END — PASTE EVERYTHING ABOVE BACK TO CLAUDE ====="

# Overall exit code: non-zero if any hard check failed (informational for the
# user / any wrapper; the EXIT trap still cleans up the background server).
if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi
exit 0
