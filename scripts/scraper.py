"""
scraper.py
----------
Fetches targeted sections of the Merck Manual Professional Edition
and saves clean text to data/raw/.

Respects robots.txt:
  - Crawl-delay: 5 seconds between requests
  - Only fetches from allowed paths (no /monograph/, /multimedia/, etc.)

Usage:
    conda activate vllm-clinical-rag
    python scripts/scraper.py

Output:
    data/raw/{topic_slug}.txt  — one file per topic
    data/raw/manifest.json     — metadata for all scraped topics
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://www.merckmanuals.com"
CRAWL_DELAY = 5  # seconds — required by robots.txt
OUTPUT_DIR = Path("data/raw")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

HEADERS = {
    "User-Agent": (
        "vllm-clinical-rag/1.0 (educational demo; "
        "github.com/federicorubiano/vllm-clinical-rag)"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Topic list ───────────────────────────────────────────────────────────────
# Curated sections covering the 5 benchmark queries + adjacent clinical topics.
# Extend this list to broaden the knowledge base.

TOPICS = [
    # ── Critical Care ────────────────────────────────────────────────────────
    {
        "slug": "sepsis-and-septic-shock",
        "url": "/professional/critical-care-medicine/sepsis-and-septic-shock/sepsis-and-septic-shock",
        "section": "Critical Care",
    },
    {
        "slug": "shock",
        "url": "/professional/critical-care-medicine/shock-and-fluid-resuscitation/shock",
        "section": "Critical Care",
    },
    {
        "slug": "acute-respiratory-distress-syndrome",
        "url": "/professional/critical-care-medicine/respiratory-failure-and-mechanical-ventilation/overview-of-mechanical-ventilation",
        "section": "Critical Care",
    },

    # ── Gastrointestinal ─────────────────────────────────────────────────────
    {
        "slug": "appendicitis",
        "url": "/professional/gastrointestinal-disorders/acute-abdomen-and-surgical-gastroenterology/appendicitis",
        "section": "Gastrointestinal",
    },

    # ── Neurology / Traumatic Brain Injury ───────────────────────────────────
    {
        "slug": "traumatic-brain-injury",
        "url": "/professional/injuries-poisoning/traumatic-brain-injury-tbi/traumatic-brain-injury-tbi",
        "section": "Neurology",
    },

    # ── Dermatology ──────────────────────────────────────────────────────────
    {
        "slug": "alopecia-areata",
        "url": "/professional/dermatologic-disorders/hair-disorders/alopecia-areata",
        "section": "Dermatology",
    },

    # ── Orthopedics ──────────────────────────────────────────────────────────
    {
        "slug": "compartment-syndrome",
        "url": "/professional/injuries-poisoning/fractures/compartment-syndrome",
        "section": "Orthopedics",
    },
]

# ── Scraper ──────────────────────────────────────────────────────────────────

def extract_text(html: str) -> str:
    """Extract clean article text from a Merck Manual topic page."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove nav, footer, sidebars, cookie banners, ads
    for tag in soup.select(
        "nav, footer, header, aside, script, style, "
        "[class*='cookie'], [class*='banner'], [class*='ad-'], "
        "[class*='sidebar'], [class*='share'], [class*='quiz'], "
        "[class*='calculator'], [class*='video'], [class*='multimedia']"
    ):
        tag.decompose()

    # Target the main article content
    main = soup.select_one("main, article, [class*='topic-content'], [id*='content']")
    if not main:
        main = soup.body

    raw = main.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in raw.splitlines()]
    lines = [line for line in lines if len(line) > 1]
    text = "\n".join(lines)

    # Remove repeated whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def scrape_topic(topic: dict, session: requests.Session) -> dict | None:
    """Fetch a single topic and save to disk. Returns metadata or None on failure."""
    url = urljoin(BASE_URL, topic["url"])
    slug = topic["slug"]
    out_path = OUTPUT_DIR / f"{slug}.txt"

    if out_path.exists():
        log.info(f"[skip] {slug} — already scraped")
        return {"slug": slug, "url": url, "section": topic["section"], "status": "cached"}

    log.info(f"[fetch] {slug}")
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"[error] {slug}: {e}")
        return {"slug": slug, "url": url, "section": topic["section"], "status": "error", "error": str(e)}

    text = extract_text(resp.text)

    if len(text) < 200:
        log.warning(f"[warn] {slug}: suspiciously short content ({len(text)} chars)")

    out_path.write_text(text, encoding="utf-8")
    log.info(f"[saved] {slug} → {out_path} ({len(text):,} chars)")

    return {
        "slug": slug,
        "url": url,
        "section": topic["section"],
        "status": "ok",
        "chars": len(text),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    session = requests.Session()

    log.info(f"Scraping {len(TOPICS)} topics with {CRAWL_DELAY}s crawl delay...")

    for i, topic in enumerate(TOPICS):
        result = scrape_topic(topic, session)
        if result:
            manifest.append(result)

        # Respect crawl delay — skip on last item
        if i < len(TOPICS) - 1:
            time.sleep(CRAWL_DELAY)

    # Save manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"\nDone. {len(manifest)} topics processed.")
    log.info(f"Manifest saved to {MANIFEST_PATH}")

    ok = sum(1 for r in manifest if r.get("status") == "ok")
    cached = sum(1 for r in manifest if r.get("status") == "cached")
    errors = sum(1 for r in manifest if r.get("status") == "error")
    log.info(f"Results: {ok} fetched, {cached} cached, {errors} errors")


if __name__ == "__main__":
    main()
