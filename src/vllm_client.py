"""
vllm_client.py
--------------
Thin wrapper around the vLLM OpenAI-compatible inference endpoint.
Handles prompt construction, citation formatting, and the
medical disclaimer appended to every response.
"""

import os
import logging
from openai import OpenAI

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


class VLLMClient:
    """
    OpenAI-compatible client pointing at a local vLLM server.

    Parameters
    ----------
    base_url  : vLLM server URL (default from env VLLM_BASE_URL)
    model     : model ID loaded in vLLM (default from env VLLM_MODEL)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        self.model    = model    or os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

        # vLLM doesn't require a real API key but the client needs a non-empty string
        self.client = OpenAI(api_key="vllm-local", base_url=self.base_url)
        log.info(f"vLLM client → {self.base_url} | model: {self.model}")

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
            sources     : list — list of {slug, section, url} dicts for the used chunks
            model       : str  — model ID
            usage       : dict — token usage from the API
        """
        user_message = build_user_prompt(question, chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
        except Exception as e:
            log.error(f"vLLM inference error: {e}")
            raise

        answer_text = response.choices[0].message.content.strip()
        answer_with_disclaimer = answer_text + DISCLAIMER

        sources = [
            {"slug": c.slug, "section": c.section, "url": c.url}
            for c in chunks
        ]

        return {
            "answer": answer_with_disclaimer,
            "sources": sources,
            "model": self.model,
            "usage": {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            },
        }
