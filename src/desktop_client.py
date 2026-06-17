"""
desktop_client.py
-----------------
Thin wrapper around Anaconda Desktop's local model server chat completions
endpoint. Uses requests (Anaconda main) — no external API keys, no third-party
SDKs, no HuggingFace Hub.

Anaconda Desktop exposes an OpenAI-compatible API at localhost:8080.
"""

import os
import logging
import requests

log = logging.getLogger(__name__)

DISCLAIMER = (
    "\n\n---\n"
    "⚠️  **Medical disclaimer:** This response is generated from the "
    "Merck Manual Professional Edition for educational and demonstration "
    "purposes only. It is not medical advice. Always consult a qualified "
    "healthcare professional for clinical decisions."
)

SYSTEM_PROMPT = """You are a clinical knowledge assistant grounded in the Merck Manual Professional Edition.

Rules you must follow without exception:
1. Answer ONLY using information present in the provided context.
2. If the context does not contain enough information to answer, say: "The provided context does not cover this query sufficiently."
3. Structure your answer with clear headings relevant to the question type:
   - For protocols: use numbered steps
   - For diagnoses: use Symptoms / Diagnosis / Treatment sections
   - For drug questions: include drug names, dosages, and indications
4. End your answer with a CITATIONS section listing each source used, in this format:
   [Section Name] — merckmanuals.com{url}
5. Do NOT add information from your training data. Stick to the context.
"""


def build_user_prompt(question: str, chunks) -> str:
    """Construct the user turn with retrieved context and question."""
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i} | {chunk.section} | {chunk.slug}]\n{chunk.text}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""###Context
{context_str}

###Question
{question}"""


class DesktopClient:
    """
    HTTP client pointing at Anaconda Desktop's local model server.
    Calls the /v1/chat/completions endpoint via requests.

    Desktop exposes an OpenAI-compatible API — no SDK needed, no external calls.

    Parameters
    ----------
    base_url  : Desktop model server URL (default from env DESKTOP_API_URL)
    model     : model name as shown in Desktop catalog (default from env INFERENCE_MODEL)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("DESKTOP_API_URL", "http://localhost:8080/v1")
        ).rstrip("/")
        self.model = model or os.getenv("INFERENCE_MODEL", "Qwen3-8B")
        log.info(f"Desktop client → {self.base_url} | model: {self.model}")

    def generate(
        self,
        question: str,
        chunks,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict:
        """
        Generate a grounded clinical answer with citations.

        Returns
        -------
        dict with keys:
            answer      : str  — full response text including citations + disclaimer
            sources     : list — list of {slug, section, url} dicts
            model       : str  — model ID
            usage       : dict — token usage from the API
        """
        user_message = build_user_prompt(question, chunks)

        payload = {
            "model":      self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       0.9,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=240,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            log.error(
                f"Cannot reach Anaconda Desktop at {self.base_url}. "
                "Is the inference model server running?"
            )
            raise
        except Exception as e:
            log.error(f"Desktop inference error: {e}")
            raise

        answer_text = data["choices"][0]["message"]["content"].strip()
        answer_with_disclaimer = answer_text + DISCLAIMER

        usage = data.get("usage", {})
        sources = [
            {"slug": c.slug, "section": c.section, "url": c.url}
            for c in chunks
        ]

        return {
            "answer":  answer_with_disclaimer,
            "sources": sources,
            "model":   self.model,
            "usage": {
                "prompt_tokens":     usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens":      usage.get("total_tokens", 0),
            },
        }
