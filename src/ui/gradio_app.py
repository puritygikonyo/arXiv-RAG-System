"""
Gradio chat UI for the Phase 7 agentic RAG system.

Runs as its own process, separate from the FastAPI backend, and talks to
POST /api/v1/ask over plain HTTP -- exactly like the curl test did. This
mirrors how you'd deploy it for real in Phase 10 (Gradio on Hugging Face
Spaces, calling a separately hosted API), so nothing needs to change later.

Run (with the FastAPI server already running on localhost:8000):
    uv run python src/ui/gradio_app.py

Then open http://localhost:7860
"""

import json

import gradio as gr
import httpx

from src.config import get_settings

settings = get_settings()

# api_host is "0.0.0.0" (a bind address, not a reachable address) --
# use localhost to actually connect to it from this separate process.
_host = "localhost" if settings.api_host == "0.0.0.0" else settings.api_host
API_URL = f"http://{_host}:{settings.api_port}/api/v1/ask"

# friendly labels shown while the graph is working, before the final answer
NODE_LABELS = {
    "guardrail": "Checking if this is answerable from the paper database...",
    "retriever": "Searching papers...",
    "grader": "Grading relevance of results...",
    "rewriter": "Refining the search query and trying again...",
    "generator": "Writing the answer...",
    "reject": "That's off-topic for this database.",
}


def _format_progress(events: list[dict]) -> str:
    """Render the running list of progress events as a small status trail."""
    lines = []
    for e in events:
        label = NODE_LABELS.get(e["node"], e["node"])
        if e["node"] == "grader":
            label += f" (relevance: {e.get('avg_relevance', 0):.2f})"
        lines.append(f"_{label}_")
    return "\n\n".join(lines)


def ask_agent(message: str, history: list[dict]):
    """
    Generator function for gr.ChatInterface (type='messages').
    Each yield REPLACES the bot's message so far -- Gradio's streaming
    convention -- so we show a running progress trail, then swap it for
    the final answer once the generator node completes.
    """
    events: list[dict] = []

    try:
        with httpx.stream(
            "POST", API_URL, json={"question": message}, timeout=90.0
        ) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                payload = json.loads(line[len("data: "):])
                node = payload.get("node")

                if node == "done":
                    break

                if node == "error":
                    yield f"Something went wrong: {payload.get('detail', 'unknown error')}"
                    return

                events.append(payload)

                if node in ("generator", "reject"):
                    answer = payload.get("answer", "")
                    citations = payload.get("citations", [])
                    if citations:
                        links = ", ".join(
                            f"[{c}](https://arxiv.org/abs/{c})" for c in citations
                        )
                        answer += f"\n\n**Sources:** {links}"
                    yield answer
                    return

                yield _format_progress(events)   # still in progress

    except httpx.ConnectError:
        yield (
            "Can't reach the API server. Make sure it's running:\n\n"
            "`uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`"
        )
    except httpx.TimeoutException:
        yield "The request timed out. Try again or check the server logs."


demo = gr.ChatInterface(
    fn=ask_agent,
    title="arXiv Research Assistant",
    description=(
        "Ask about papers in the index. The agent searches, grades "
        "relevance, and automatically rewrites the query if the first "
        "search doesn't find a good match."
    ),
    examples=[
        "How does the transformer architecture work?",
        "What is the attention mechanism?",
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)