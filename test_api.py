"""
test_api.py
-----------
Quick connectivity test for the Clinical Knowledge API.
Run this after starting the API to verify everything is wired up correctly.

Usage:
    python test_api.py
    python test_api.py --url http://localhost:8000

You should see green checkmarks for all tests.
"""

import argparse
import os
import sys

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


def check(label: str, passed: bool, detail: str = ""):
    symbol = "✅" if passed else "❌"
    print(f"  {symbol}  {label}", end="")
    if detail:
        print(f"  ({detail})", end="")
    print()
    return passed


def main(api_url: str):
    print()
    print("Clinical Knowledge API — connectivity test")
    print(f"Target: {api_url}")
    print("-" * 50)

    all_passed = True

    # ── 1. Health check ───────────────────────────────────────────────────────
    print("\n[1] Health check")
    try:
        r = requests.get(f"{api_url}/health", timeout=10)
        health = r.json()
        all_passed &= check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
        all_passed &= check("Status OK", health.get("status") == "ok")
        all_passed &= check(
            "Chunks loaded",
            health.get("chunks_loaded", 0) > 0,
            f"{health.get('chunks_loaded', 0)} chunks",
        )
        all_passed &= check("Model set", bool(health.get("model")), health.get("model", "—"))
    except requests.exceptions.ConnectionError:
        check("API reachable", False, f"cannot connect to {api_url}")
        print()
        print("  Is the API running? Start it with:")
        print("    uvicorn src.api:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    # ── 2. Query test ─────────────────────────────────────────────────────────
    print("\n[2] Query test")
    TEST_QUESTION = "What is the protocol for managing sepsis in a critical care unit?"
    data = None   # ensure it's defined even if the request below times out / errors
    try:
        r = requests.post(
            f"{api_url}/query",
            json={"question": TEST_QUESTION, "max_tokens": 256},
            timeout=240,
        )
        data = r.json()
        all_passed &= check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
        all_passed &= check("Answer returned", bool(data.get("answer")))
        all_passed &= check(
            "Sources returned",
            len(data.get("sources", [])) > 0,
            f"{len(data.get('sources', []))} sources",
        )
        all_passed &= check(
            "Latency reported",
            data.get("latency_ms", 0) > 0,
            f"{data.get('latency_ms', '—')} ms",
        )
    except Exception as e:
        check("Query succeeded", False, str(e))
        all_passed = False

    # ── 3. Disclaimer check ───────────────────────────────────────────────────
    print("\n[3] Safety checks")
    if data:
        answer = data.get("answer", "")
        has_disclaimer = any(
            phrase in answer.lower()
            for phrase in ["medical disclaimer", "not medical advice", "educational and demonstration"]
        )
        all_passed &= check("Medical disclaimer present in answer", has_disclaimer)
        all_passed &= check(
            "Answer length reasonable",
            50 < len(answer) < 10000,
            f"{len(answer)} chars",
        )

    # ── 4. Validation check ───────────────────────────────────────────────────
    print("\n[4] Input validation")
    try:
        r = requests.post(
            f"{api_url}/query",
            json={"question": "hi"},  # too short — should fail validation
            timeout=10,
        )
        all_passed &= check(
            "Short query rejected (422)",
            r.status_code == 422,
            f"got {r.status_code}",
        )
    except Exception as e:
        check("Validation test", False, str(e))
        all_passed = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("-" * 50)
    if all_passed:
        print("  ✅  All tests passed — API is ready!")
    else:
        print("  ❌  Some tests failed. Check the output above.")
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Clinical Knowledge API")
    parser.add_argument(
        "--url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="Base URL of the API (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    main(args.url)
