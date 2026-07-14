"""
Ask router — exposes the Phase 7 LangGraph agent over HTTP.

POST /api/v1/ask streams progress from the agent graph as Server-Sent
Events (SSE): one event per node as it completes, so the client can show
"searching -> grading -> generating" instead of waiting silently for the
whole chain (guardrail + up to N retrieval/grade/rewrite loops +
generation) to finish, which can take 15-40+ seconds end to end.

Event shapes (one JSON object per SSE `data:` line):
    {"node": "guardrail",  "is_on_topic": true}
    {"node": "retriever",  "chunks_found": 3, "attempt": 1}
    {"node": "grader",     "avg_relevance": 0.82}
    {"node": "rewriter",   "rewritten_query": "..."}
    {"node": "generator",  "status": "answered", "answer": "...", "citations": [...]}
    {"node": "reject",     "status": "off_topic", "answer": "..."}
    {"node": "done"}
    {"node": "error",      "detail": "..."}

NOTE: this streams graph PROGRESS (which node just finished), not
token-by-token generation of the answer text itself. The full answer
arrives in one event when the generator node completes. True token
streaming needs LangGraph's astream_events API instead — a reasonable
follow-up once this is working end to end.
"""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.logger import get_logger
from src.schemas.ask import AskRequest
from src.services.agents.workflow import agent_graph

router = APIRouter(tags=["ask"])
logger = get_logger(__name__)


def _sse(payload: dict) -> str:
    """Format a dict as one Server-Sent Event line."""
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_agent(question: str) -> AsyncGenerator[str, None]:
    try:
        # stream_mode="updates" yields {node_name: {partial_state_changes}}
        # once per completed node -- NOT the default mode, which yields
        # the full accumulated state each time.
        async for step in agent_graph.astream(
            {"query": question}, stream_mode="updates"
        ):
            for node_name, update in step.items():
                event: dict = {"node": node_name}

                if node_name == "guardrail":
                    event["is_on_topic"] = update.get("is_on_topic")
                elif node_name == "retriever":
                    event["chunks_found"] = len(update.get("chunks", []))
                    event["attempt"] = update.get("retrieval_attempts")
                elif node_name == "grader":
                    event["avg_relevance"] = round(update.get("avg_relevance", 0.0), 3)
                elif node_name == "rewriter":
                    event["rewritten_query"] = update.get("rewritten_query")
                elif node_name in ("generator", "reject"):
                    event["status"] = update.get("status")
                    event["answer"] = update.get("answer")
                    event["citations"] = update.get("citations", [])

                logger.info("ask_stream_event", **event)
                yield _sse(event)

        yield _sse({"node": "done"})

    except Exception as exc:
        logger.error("ask_stream_failed", error=str(exc), question=question[:100])
        yield _sse({"node": "error", "detail": str(exc)})


@router.post(
    "/ask",
    summary="Ask the research agent a question",
    description=(
        "Runs the full LangGraph agentic RAG pipeline (guardrail -> "
        "retrieval -> grading -> optional rewrite loop -> generation) "
        "and streams progress plus the final answer as Server-Sent Events."
    ),
)
async def ask(request: AskRequest) -> StreamingResponse:
    logger.info("ask_request", question=request.question[:100])

    return StreamingResponse(
        _stream_agent(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # disable proxy buffering (relevant if ever deployed behind
            # nginx/HF Spaces proxy in Phase 10) so events arrive as they're
            # generated instead of being batched
            "X-Accel-Buffering": "no",
        },
    )