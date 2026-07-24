"""
Ask router — exposes the Phase 7 LangGraph agent over HTTP.

POST /api/v1/ask streams progress from the agent graph as Server-Sent
Events (SSE): one event per node as it completes, so the client can show
"searching -> grading -> generating" instead of waiting silently for the
whole chain (guardrail + up to N retrieval/grade/rewrite loops +
generation) to finish, which can take 15-40+ seconds end to end.

Phase 8: checks the semantic cache before running the graph. On a hit,
returns the cached answer immediately (skips guardrail/retrieval/grading/
generation entirely) and logs a lightweight manual Langfuse trace tagged
"cache_hit" so hit rate is visible in the dashboard even though no graph
nodes ran. On a miss, runs the full graph as before (tagged "cache_miss"
inside guardrail_node) and stores the result afterward.

Invite enforcement (added alongside the Gradio web UI): every request
carries an invite_token. Before the cache check or any graph work runs,
we verify the token is valid, not revoked, and under its daily limit —
a blocked/exhausted user costs nothing, not even a cache lookup.

Event shapes (one JSON object per SSE `data:` line):
    {"node": "blocked",    "reason": "..."}
    {"node": "cache",      "cache_hit": true, "similarity": 0.94}
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
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.database import AsyncSessionLocal
from src.logger import get_logger
from src.models import QueryLog
from src.schemas.ask import AskRequest
from src.services.agents.workflow import agent_graph
from src.services.cache.semantic_cache import get_cached_answer, set_cached_answer
from src.services.invite_check import check_invite_allowed
from src.services.monitoring.langfuse_client import get_langfuse_client

router = APIRouter(tags=["ask"])
logger = get_logger(__name__)


def _sse(payload: dict) -> str:
    """Format a dict as one Server-Sent Event line."""
    return f"data: {json.dumps(payload)}\n\n"


async def _log_query(
    query: str,
    cache_hit: bool,
    latency_ms: int,
    status: str,
    invite_token: str | None = None,
) -> None:
    """
    Write one row to query_logs. Uses its own short-lived session rather
    than FastAPI's Depends(get_db) — this function runs inside a
    StreamingResponse generator, outside the normal request/response
    dependency-injection lifecycle, so it needs to open and close its
    own session explicitly.

    Failures here are logged but never raised — a broken metrics write
    should never take down an otherwise-successful user-facing response.
    """
    try:
        async with AsyncSessionLocal() as session:
            session.add(QueryLog(
                query=query,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                status=status,
                invite_token=invite_token,
            ))
            await session.commit()
    except Exception as exc:
        logger.warning("query_log_write_failed", error=str(exc))


def _log_cache_hit_trace(question: str, answer: str, citations: list[str]) -> None:
    """
    Manually log a trace for a cache hit.
    ...
    [unchanged from before]
    """
    try:
        client = get_langfuse_client()
        trace = client.trace(
            name="agentic_rag_query",
            input={"query": question},
            tags=["arxiv-rag", "cache_hit"],
        )
        trace.update(output={"answer": answer, "citations": citations})
    except Exception as exc:
        logger.warning("cache_hit_trace_failed", error=str(exc))


async def _stream_agent(
    question: str, invite_token: str | None
) -> AsyncGenerator[str, None]:
    request_start = time.monotonic()

    # ── Step 0: invite enforcement — before anything else runs ──────────
    allowed, reason = await check_invite_allowed(invite_token)
    if not allowed:
        logger.info("ask_blocked", reason=reason, invite_token=invite_token)
        yield _sse({"node": "blocked", "reason": reason})
        yield _sse({"node": "done"})
        return

    # ── Step 1: check the semantic cache before running anything ────────
    cached = await get_cached_answer(question)

    if cached is not None:
        answer = cached["answer"]
        citations = cached.get("citations", [])

        logger.info("ask_cache_hit", question=question[:100])
        _log_cache_hit_trace(question, answer, citations)

        yield _sse({"node": "cache", "cache_hit": True})
        yield _sse({
            "node": "generator",
            "status": "answered",
            "answer": answer,
            "citations": citations,
        })

        latency_ms = int((time.monotonic() - request_start) * 1000)
        await _log_query(
            question, cache_hit=True, latency_ms=latency_ms,
            status="answered", invite_token=invite_token,
        )

        yield _sse({"node": "done"})
        return

    # ── Step 2: cache miss — run the full agent graph ───────────────────
    logger.info("ask_cache_miss", question=question[:100])
    final_answer: str | None = None
    final_citations: list[str] = []
    final_status: str = "error"

    try:
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
                    final_status = update.get("status", "error")
                    if update.get("status") == "answered":
                        final_answer = update.get("answer")
                        final_citations = update.get("citations", [])

                logger.info("ask_stream_event", **event)
                yield _sse(event)

        if final_answer:
            await set_cached_answer(question, final_answer, final_citations)

        latency_ms = int((time.monotonic() - request_start) * 1000)
        await _log_query(
            question, cache_hit=False, latency_ms=latency_ms,
            status=final_status, invite_token=invite_token,
        )

        yield _sse({"node": "done"})

    except Exception as exc:
        logger.error("ask_stream_failed", error=str(exc), question=question[:100])

        latency_ms = int((time.monotonic() - request_start) * 1000)
        await _log_query(
            question, cache_hit=False, latency_ms=latency_ms,
            status="error", invite_token=invite_token,
        )

        yield _sse({"node": "error", "detail": str(exc)})


@router.post(
    "/ask",
    summary="Ask the research agent a question",
    description=(
        "Runs the full LangGraph agentic RAG pipeline (guardrail -> "
        "retrieval -> grading -> optional rewrite loop -> generation) "
        "and streams progress plus the final answer as Server-Sent Events. "
        "Checks a semantic cache first and returns immediately on a hit."
    ),
)
async def ask(request: AskRequest) -> StreamingResponse:
    logger.info("ask_request", question=request.question[:100])

    return StreamingResponse(
        _stream_agent(request.question, request.invite_token),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )