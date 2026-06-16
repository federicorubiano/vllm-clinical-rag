"""
eval/run_eval.py
----------------
Evaluation harness for the Clinical Knowledge API.
Scores all 5 benchmark queries across five dimensions:
  - Groundedness   : Does the answer stay within the retrieved context?
  - Relevance      : Does the answer address the question asked?
  - Citation Rate  : Fraction of responses that include at least one citation
  - Disclaimer     : Is the mandatory medical disclaimer present?
  - Structure      : Is the response well-formatted with headings/steps?

V1 failure mode addressed: self-judging (Mistral scoring its own outputs)
V2 fix: custom heuristic scoring — no LLM-as-judge, fully reproducible

All packages used are available on Anaconda main channel.

Usage:
    # API must be running first:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

    # Run evaluation:
    python eval/run_eval.py
    python eval/run_eval.py --api-url http://localhost:8000 --output eval/results.json
    python eval/run_eval.py --report   # also generates eval/report.html

Output:
    eval/results.json     — per-query scores + aggregate metrics
    eval/report.html      — Evidently AI report (open in browser)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Benchmark queries (same as Gradio demo buttons) ──────────────────────────

BENCHMARK_QUERIES = [
    {
        "id": "sepsis_protocol",
        "question": "What is the protocol for managing sepsis in a critical care unit?",
        "expected_sections": ["sepsis", "critical care", "treatment"],
    },
    {
        "id": "appendicitis",
        "question": "What are the common symptoms for appendicitis, and can it be cured via medicine?",
        "expected_sections": ["appendicitis", "symptoms", "treatment"],
    },
    {
        "id": "alopecia",
        "question": "What are the effective treatments for sudden patchy hair loss on the scalp?",
        "expected_sections": ["alopecia", "hair loss", "treatment"],
    },
    {
        "id": "tbi",
        "question": "What treatments are recommended for traumatic brain injury?",
        "expected_sections": ["traumatic brain injury", "tbi", "treatment"],
    },
    {
        "id": "fracture_hiking",
        "question": "What are the precautions and treatment steps for a leg fracture during a hiking trip?",
        "expected_sections": ["fracture", "orthopedic", "treatment"],
    },
]


# ── API client ────────────────────────────────────────────────────────────────

def call_api(api_url: str, question: str, max_tokens: int = 512) -> dict | None:
    """Call the Clinical Knowledge API and return the parsed response."""
    try:
        resp = requests.post(
            f"{api_url}/query",
            json={"question": question, "max_tokens": max_tokens, "temperature": 0.1},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot reach API at {api_url}. Is uvicorn running?")
        return None
    except Exception as e:
        log.error(f"API call failed: {e}")
        return None


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_citation_rate(answer: str, sources: list) -> float:
    """
    1.0 if the answer references at least one source in its CITATIONS section,
    0.5 if sources were returned but not referenced in text,
    0.0 if no sources returned at all.
    """
    if not sources:
        return 0.0
    has_citation_section = "CITATIONS" in answer.upper() or "merckmanuals.com" in answer
    return 1.0 if has_citation_section else 0.5


def score_disclaimer_present(answer: str) -> float:
    """1.0 if the mandatory medical disclaimer is present."""
    disclaimer_markers = ["medical disclaimer", "not medical advice", "educational and demonstration"]
    return 1.0 if any(m in answer.lower() for m in disclaimer_markers) else 0.0


def score_structure(answer: str, query_id: str) -> float:
    """
    Heuristic: well-structured answers have headings or numbered steps.
    Protocol queries should have numbered steps; diagnosis queries should have sections.
    """
    has_heading   = any(line.startswith("#") for line in answer.splitlines())
    has_numbered  = any(line.strip()[:2].rstrip(".").isdigit() for line in answer.splitlines())
    has_bold      = "**" in answer

    score = 0.0
    if has_heading or has_bold:
        score += 0.5
    if has_numbered:
        score += 0.5
    return min(score, 1.0)


def score_groundedness_heuristic(answer: str, sources: list) -> float:
    """
    Lightweight groundedness proxy: checks how many source section names
    appear in the answer text. Full Evidently groundedness uses LLM-as-judge;
    this is the offline fallback that requires no extra model call.

    Returns a float in [0, 1].
    """
    if not sources:
        return 0.0

    hits = sum(
        1 for s in sources
        if s.get("section", "").lower() in answer.lower()
    )
    return round(hits / len(sources), 3)


def score_relevance_heuristic(question: str, answer: str, expected_sections: list[str]) -> float:
    """
    Proxy for relevance: checks whether the answer addresses the key topics
    from the question. Full Evidently relevance uses semantic similarity;
    this is the offline fallback.
    """
    q_tokens = set(question.lower().split())
    a_tokens = set(answer.lower().split())
    keyword_overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)

    section_hits = sum(1 for s in expected_sections if s.lower() in answer.lower())
    section_score = section_hits / max(len(expected_sections), 1)

    return round((keyword_overlap + section_score) / 2, 3)


# ── HTML report ───────────────────────────────────────────────────────────────

def _basic_html_report(results: list[dict], output_dir: Path) -> str | None:
    """
    Fallback HTML report using only pandas (Anaconda main channel).
    Used when Evidently is unavailable so the harness never hard-fails.
    """
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "Query":          r["question"][:80] + ("…" if len(r["question"]) > 80 else ""),
            "Groundedness":   r["scores"]["groundedness"],
            "Relevance":      r["scores"]["relevance"],
            "Citation Rate":  r["scores"]["citation_rate"],
            "Disclaimer":     r["scores"]["disclaimer_present"],
            "Structure":      r["scores"]["structure"],
            "Overall":        r["scores"]["overall"],
            "Latency (ms)":   r["latency_ms"],
        })

    df = pd.DataFrame(rows)

    def bar(val: float) -> str:
        filled = int(val * 10)
        return "█" * filled + "░" * (10 - filled)

    rows_html = ""
    for _, row in df.iterrows():
        rows_html += "<tr>" + "".join(
            f"<td>{v:.2f} {bar(v)}</td>" if isinstance(v, float) and k != "Latency (ms)"
            else f"<td>{v}</td>"
            for k, v in row.items()
        ) + "</tr>\n"

    agg = df.select_dtypes(include="number").mean().round(3)
    agg_html = "<tr><td><strong>Average</strong></td>" + "".join(
        f"<td><strong>{v:.2f} {bar(v)}</strong></td>" if k != "Latency (ms)"
        else f"<td><strong>{v:.0f}</strong></td>"
        for k, v in agg.items()
    ) + "</tr>"

    headers = "".join(f"<th>{c}</th>" for c in df.columns)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Clinical RAG — Evaluation Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
    p.meta {{ color: #666; font-size: 0.85rem; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
    th {{ background: #f0f0f0; padding: 8px 12px; text-align: left; border-bottom: 2px solid #ccc; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
    tr:last-child td {{ border-bottom: 2px solid #ccc; font-weight: bold; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>🏥 Clinical Knowledge RAG — Evaluation Report</h1>
  <p class="meta">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · {len(rows)} queries · All packages from Anaconda main channel</p>
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}{agg_html}</tbody>
  </table>
  <p style="margin-top:24px;font-size:0.8rem;color:#999;">
    ⚠️ For educational and demonstration purposes only. Nothing in this report constitutes medical advice.
  </p>
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    log.info(f"Report saved → {report_path}")
    return str(report_path)


# ── Evidently report (primary) ────────────────────────────────────────────────

def run_report(results: list[dict], output_dir: Path) -> str | None:
    """
    Generate the HTML evaluation report with **Evidently AI** (Anaconda main).

    Builds an Evidently Dataset from the per-query scores and renders a
    DataSummaryPreset report — descriptive statistics across every metric
    column. No LLM-as-judge, no reference dataset, fully offline.

    Falls back to a minimal pandas table (`_basic_html_report`) if Evidently
    is unavailable, so the reporting step never hard-fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"

    rows = [
        {
            "query":         r["question"][:80],
            "groundedness":  r["scores"]["groundedness"],
            "relevance":     r["scores"]["relevance"],
            "citation_rate": r["scores"]["citation_rate"],
            "disclaimer":    r["scores"]["disclaimer_present"],
            "structure":     r["scores"]["structure"],
            "overall":       r["scores"]["overall"],
            "latency_ms":    r["latency_ms"],
        }
        for r in results
    ]

    try:
        import pandas as pd
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataSummaryPreset

        df = pd.DataFrame(rows)
        dataset = Dataset.from_pandas(df, data_definition=DataDefinition())
        report = Report([DataSummaryPreset()])
        result = report.run(dataset, None)   # single dataset, no reference
        result.save_html(str(report_path))
        log.info(f"Evidently report saved → {report_path}")
        return str(report_path)
    except Exception as e:
        log.warning(f"Evidently unavailable ({e}); using basic HTML table instead.")
        return _basic_html_report(results, output_dir)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(api_url: str, output_path: Path) -> dict:
    """
    Run all benchmark queries, score each response, and return aggregated results.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify API is up
    try:
        health = requests.get(f"{api_url}/health", timeout=10).json()
        log.info(f"API healthy — model: {health['model']}, chunks: {health['chunks_loaded']}")
    except Exception as e:
        log.error(f"Health check failed: {e}")
        sys.exit(1)

    results = []
    log.info(f"Running {len(BENCHMARK_QUERIES)} benchmark queries...")

    for i, bq in enumerate(BENCHMARK_QUERIES, 1):
        log.info(f"[{i}/{len(BENCHMARK_QUERIES)}] {bq['id']}: {bq['question'][:60]}...")

        t0 = time.perf_counter()
        response = call_api(api_url, bq["question"])
        elapsed = int((time.perf_counter() - t0) * 1000)

        if response is None:
            log.warning(f"  ✗ No response for {bq['id']}")
            results.append({
                "id": bq["id"],
                "question": bq["question"],
                "answer": "",
                "sources": [],
                "latency_ms": elapsed,
                "scores": {
                    "groundedness":    0.0,
                    "relevance":       0.0,
                    "citation_rate":   0.0,
                    "disclaimer_present": 0.0,
                    "structure":       0.0,
                },
                "error": "no_response",
            })
            continue

        answer  = response.get("answer", "")
        sources = response.get("sources", [])

        scores = {
            "groundedness":       score_groundedness_heuristic(answer, sources),
            "relevance":          score_relevance_heuristic(bq["question"], answer, bq["expected_sections"]),
            "citation_rate":      score_citation_rate(answer, sources),
            "disclaimer_present": score_disclaimer_present(answer),
            "structure":          score_structure(answer, bq["id"]),
        }

        overall = round(
            0.30 * scores["groundedness"]
            + 0.30 * scores["relevance"]
            + 0.20 * scores["citation_rate"]
            + 0.10 * scores["disclaimer_present"]
            + 0.10 * scores["structure"],
            3,
        )
        scores["overall"] = overall

        log.info(
            f"  ✓ overall={overall:.2f} | ground={scores['groundedness']:.2f} | "
            f"rel={scores['relevance']:.2f} | cite={scores['citation_rate']:.2f} | "
            f"latency={response.get('latency_ms', elapsed)}ms"
        )

        results.append({
            "id":          bq["id"],
            "question":    bq["question"],
            "answer":      answer,
            "sources":     sources,
            "model":       response.get("model", "unknown"),
            "latency_ms":  response.get("latency_ms", elapsed),
            "usage":       response.get("usage", {}),
            "scores":      scores,
        })

    # Aggregate metrics
    scored = [r for r in results if "error" not in r]
    aggregate = {}
    if scored:
        for metric in ["groundedness", "relevance", "citation_rate", "disclaimer_present", "structure", "overall"]:
            vals = [r["scores"][metric] for r in scored]
            aggregate[metric] = round(sum(vals) / len(vals), 3)

        aggregate["avg_latency_ms"] = round(
            sum(r["latency_ms"] for r in scored) / len(scored)
        )
        aggregate["queries_scored"] = len(scored)
        aggregate["queries_failed"] = len(results) - len(scored)

    output = {
        "eval_timestamp": datetime.utcnow().isoformat() + "Z",
        "api_url":        api_url,
        "model":          scored[0]["model"] if scored else "unknown",
        "aggregate":      aggregate,
        "results":        results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved → {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries scored : {aggregate.get('queries_scored', 0)}/{len(BENCHMARK_QUERIES)}")
    print(f"  Avg latency    : {aggregate.get('avg_latency_ms', '—')} ms")
    print()
    for metric in ["groundedness", "relevance", "citation_rate", "disclaimer_present", "overall"]:
        bar_len = int(aggregate.get(metric, 0) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {metric:<22} {bar}  {aggregate.get(metric, 0):.2f}")
    print("=" * 60)

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clinical Knowledge API — evaluation harness")
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="Base URL of the running FastAPI server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        default="eval/results.json",
        help="Path for the JSON results file (default: eval/results.json)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate the Evidently AI HTML report (falls back to a basic table if unavailable)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    eval_output = run_evaluation(args.api_url, output_path)

    if args.report:
        run_report(eval_output["results"], output_path.parent)


if __name__ == "__main__":
    main()
