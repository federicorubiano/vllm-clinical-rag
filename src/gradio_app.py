"""
gradio_app.py
-------------
Gradio demo UI for the Clinical Knowledge API.
Calls the FastAPI /query endpoint and renders answers with citations.

Start (after the API is running):
    python src/gradio_app.py
"""

import os
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000")

BENCHMARK_QUERIES = [
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the common symptoms for appendicitis, and can it be cured via medicine?",
    "What are the effective treatments for sudden patchy hair loss on the scalp?",
    "What treatments are recommended for traumatic brain injury?",
    "What are the precautions and treatment steps for a leg fracture during a hiking trip?",
]

DESCRIPTION = """
## 🏥 Clinical Knowledge API — Demo
**Powered by Anaconda Desktop · FAISS · FastAPI**

Answers clinical queries grounded in the **Merck Manual Professional Edition**.
Every response includes source citations and a mandatory medical disclaimer.

> ## ⚠️ MEDICAL DISCLAIMER
>
> **This demo is for educational and demonstration purposes only.**
>
> Nothing produced by this system — including all generated text, citations, and clinical summaries — constitutes medical advice, diagnosis, or treatment. The system may produce inaccurate, incomplete, or outdated information even when citing real sources.
>
> **Always consult a qualified, licensed healthcare professional before making any clinical decision.** Do not use this tool in any real patient care setting.
"""


def query_api(question: str, max_tokens: int, temperature: float):
    """Call the FastAPI /query endpoint and format the response."""
    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return (
            "❌ Cannot connect to the API. Is `uvicorn src.api:app` running?",
            "", "",
        )
    except requests.exceptions.HTTPError as e:
        return f"❌ API error: {e.response.status_code} — {e.response.text}", "", ""
    except Exception as e:
        return f"❌ Unexpected error: {e}", "", ""

    answer = data.get("answer", "No answer returned.")

    # Format sources
    sources = data.get("sources", [])
    if sources:
        source_lines = [
            f"- **{s['section']}** — [merckmanuals.com{s['url']}](https://www.merckmanuals.com{s['url']})"
            for s in sources
        ]
        sources_text = "\n".join(source_lines)
    else:
        sources_text = "_No sources returned._"

    # Format usage stats
    usage = data.get("usage", {})
    latency = data.get("latency_ms", "—")
    model = data.get("model", "—")
    stats = (
        f"**Model:** {model}  \n"
        f"**Latency:** {latency} ms  \n"
        f"**Tokens:** {usage.get('prompt_tokens', '—')} prompt · "
        f"{usage.get('completion_tokens', '—')} completion"
    )

    return answer, sources_text, stats


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Clinical Knowledge API", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g. What is the protocol for managing sepsis?",
                lines=2,
            )
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    64, 1024, value=512, step=64, label="Max tokens"
                )
                temperature_slider = gr.Slider(
                    0.0, 1.0, value=0.1, step=0.05, label="Temperature"
                )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Benchmark queries")
            for q in BENCHMARK_QUERIES:
                gr.Button(q, size="sm").click(
                    fn=lambda x=q: x,
                    outputs=question_box,
                )

    answer_box = gr.Markdown(label="Answer")

    with gr.Row():
        sources_box = gr.Markdown(label="Sources")
        stats_box   = gr.Markdown(label="Stats")

    submit_btn.click(
        fn=query_api,
        inputs=[question_box, max_tokens_slider, temperature_slider],
        outputs=[answer_box, sources_box, stats_box],
    )

    question_box.submit(
        fn=query_api,
        inputs=[question_box, max_tokens_slider, temperature_slider],
        outputs=[answer_box, sources_box, stats_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
