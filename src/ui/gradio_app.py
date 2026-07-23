"""
Gradio chat UI for the agentic RAG system — invite-only access.

Runs as its own process, talks to POST /api/v1/ask over plain HTTP.

ACCESS CONTROL:
  Gradio's built-in auth expects (username, password). We repurpose this:
  username can be anything (their name, for your own reference in logs),
  password is their invite token (generated via generate_invite.py).
  auth_fn() checks the token against the `invites` table in Postgres —
  revoked or unknown tokens are rejected.

Run locally (with the FastAPI server already running):
    uv run python src/ui/gradio_app.py
Then open http://localhost:7860

For deployment, set GRADIO_API_URL to your deployed Render API's /api/v1/ask
endpoint instead of relying on the localhost default.
"""

import json
import os
from datetime import UTC, datetime

import gradio as gr
import httpx
from sqlalchemy import select

from src.config import get_settings
from src.database import AsyncSessionLocal
from src.models import Invite

settings = get_settings()

_host = "localhost" if settings.api_host == "0.0.0.0" else settings.api_host
DEFAULT_API_URL = f"http://{_host}:{settings.api_port}/api/v1/ask"
# Override with the deployed Render URL when running this UI as its own
# service, e.g. GRADIO_API_URL=https://arxiv-rag-system-xxxx.onrender.com/api/v1/ask
API_URL = os.environ.get("GRADIO_API_URL", DEFAULT_API_URL)

NODE_LABELS = {
    "guardrail": "Checking if this is answerable from the paper database...",
    "retriever": "Searching papers...",
    "grader": "Grading relevance of results...",
    "rewriter": "Refining the search query and trying again...",
    "generator": "Writing the answer...",
    "reject": "That's off-topic for this database.",
}


def _format_progress(events: list[dict]) -> str:
    lines = []
    for e in events:
        label = NODE_LABELS.get(e["node"], e["node"])
        if e["node"] == "grader":
            label += f" (relevance: {e.get('avg_relevance', 0):.2f})"
        lines.append(f"_{label}_")
    return "\n\n".join(lines)


async def _check_invite(password: str) -> bool:
    """
    Look up the submitted password as an invite token. Valid + not revoked
    => allow in, and stamp first_used_at / last_used_at for visibility
    into who's actually using their invite.
    """
    async with AsyncSessionLocal() as session:
        invite = await session.scalar(select(Invite).where(Invite.token == password))
        if invite is None or invite.revoked:
            return False

        now = datetime.now(UTC)
        if invite.first_used_at is None:
            invite.first_used_at = now
        invite.last_used_at = now
        await session.commit()
        return True


async def auth_fn(username: str, password: str) -> bool:
    """
    Gradio supports async auth callables directly — no need to wrap with
    asyncio.run, which can conflict with Gradio's own running event loop
    and silently fail (caught as a generic exception, misread as "wrong
    password" when it's actually a Python-level event loop conflict).
    """
    try:
        return await _check_invite(password)
    except Exception as e:
        # Log this instead of silently swallowing it — a DB hiccup during
        # login should be visible, not indistinguishable from a wrong token.
        print(f"[auth_fn] Error checking invite: {e}")
        return False


def ask_agent(message: str, history: list[dict]):
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

                yield _format_progress(events)

    except httpx.ConnectError:
        yield (
            "Can't reach the API server. Make sure it's running, or check "
            "GRADIO_API_URL is pointed at the right place."
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
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=auth_fn)