"""
Integration tests for Phase 7 — Agentic RAG (guardrail, retriever, grader,
rewriter, generator, the compiled graph, and POST /api/v1/ask).

Converted from the manual scripts/test_retriever.py, scripts/test_grader.py,
and scripts/test_agent_e2e.py debug scripts used while building each node.

Unlike test_search.py, these tests exercise REAL Groq calls and REAL
OpenSearch vector search (no mocking) -- so assertions check status codes,
response shape, and value ranges rather than exact answer text, since LLM
wording isn't deterministic. This mirrors test_search.py's pattern of
accepting multiple valid status codes rather than asserting exact data.

Requires: GROQ_API_KEY set and OpenSearch running with at least one paper
indexed via POST /api/v1/hybrid-search/embed-and-index. Tests that would
otherwise fail due to unreachable services skip gracefully rather than
falsely reporting a code bug.
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.services.agents.workflow import agent_graph
from src.services.search.client import init_opensearch


def _parse_sse_events(raw_text: str) -> list[dict]:
    """Parse a raw SSE response body into a list of event dicts."""
    events = []
    for line in raw_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


# ---------------------------------------------------------------------------
# HTTP-level tests: POST /api/v1/ask
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestAskEndpoint:
    """Integration tests for POST /api/v1/ask."""

    @pytest.fixture
    async def client(self) -> AsyncClient:  # type: ignore[override]
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            yield c

    async def test_ask_rejects_empty_question(self, client: AsyncClient) -> None:
        """Empty question should return 422 -- same validation pattern as /search."""
        resp = await client.post("/api/v1/ask", json={"question": ""})
        assert resp.status_code == 422

    async def test_ask_rejects_missing_question(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/ask", json={})
        assert resp.status_code == 422

    async def test_ask_returns_event_stream(self, client: AsyncClient) -> None:
        """A valid question always returns a 200 SSE stream -- failures
        surface as an 'error' event INSIDE the stream, not an HTTP error
        status, since streaming has already started by the time a node fails."""
        resp = await client.post(
            "/api/v1/ask", json={"question": "What is the attention mechanism?"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    async def test_ask_off_topic_question_is_rejected_by_guardrail(
        self, client: AsyncClient
    ) -> None:
        """An obviously off-topic question should reach the 'reject' node.
        Requires a working Groq key (guardrail is an LLM call) -- skips if
        Groq/OpenSearch aren't reachable rather than failing."""
        resp = await client.post(
            "/api/v1/ask", json={"question": "What's a good pizza topping?"}
        )
        events = _parse_sse_events(resp.text)
        node_names = [e["node"] for e in events]

        if "error" in node_names:
            pytest.skip("Groq/OpenSearch not reachable in this environment")

        reject_events = [e for e in events if e["node"] == "reject"]
        assert len(reject_events) == 1
        assert reject_events[0]["status"] == "off_topic"

    async def test_ask_on_topic_question_reaches_generator_or_errors_cleanly(
        self, client: AsyncClient
    ) -> None:
        """An on-topic question should reach 'generator' with status
        'answered' or 'no_relevant_docs' -- never crash unhandled. If
        Groq/OpenSearch aren't reachable, the stream should still end with
        a well-formed 'error' event, not a 500."""
        resp = await client.post(
            "/api/v1/ask",
            json={"question": "How does the transformer architecture work?"},
        )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.text)
        node_names = [e["node"] for e in events]
        assert "done" in node_names or "error" in node_names

        if "error" not in node_names:
            generator_events = [e for e in events if e["node"] == "generator"]
            assert len(generator_events) == 1
            assert generator_events[0]["status"] in ("answered", "no_relevant_docs")
            assert "answer" in generator_events[0]


# ---------------------------------------------------------------------------
# Node-level tests: individual agent nodes, bypassing HTTP
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestRetrieverNode:
    """Converted from scripts/test_retriever.py."""

    @pytest.fixture(autouse=True)
    async def _init_opensearch(self) -> None:
        await init_opensearch()

    async def test_retriever_returns_well_shaped_chunks(self) -> None:
        from src.services.agents.nodes.retriever import retriever_node

        result = await retriever_node(
            {"search_query": "gradient vanishing problem", "retrieval_attempts": 0}
        )

        assert "chunks" in result
        assert result["retrieval_attempts"] == 1
        for chunk in result["chunks"]:
            assert {"chunk_id", "paper_id", "paper_title", "text", "score"} <= chunk.keys()


@pytest.mark.integration
class TestGraderNode:
    """Converted from scripts/test_grader.py."""

    async def test_grader_scores_within_valid_range(self) -> None:
        from src.services.agents.nodes.grader import grader_node

        fake_chunks = [
            {
                "chunk_id": "test-1",
                "paper_id": "1706.03762",
                "paper_title": "Attention Is All You Need",
                "text": "The Transformer is based solely on attention mechanisms.",
                "score": 1.0,
                "relevance": 0.0,
            }
        ]

        result = await grader_node(
            {"query": "What is the attention mechanism?", "chunks": fake_chunks}
        )

        assert len(result["graded_chunks"]) == 1
        assert 0.0 <= result["avg_relevance"] <= 1.0
        assert 0.0 <= result["graded_chunks"][0]["relevance"] <= 1.0

    async def test_grader_handles_empty_chunks(self) -> None:
        from src.services.agents.nodes.grader import grader_node

        result = await grader_node({"query": "anything", "chunks": []})

        assert result["graded_chunks"] == []
        assert result["avg_relevance"] == 0.0


# ---------------------------------------------------------------------------
# Full-graph tests: the compiled LangGraph, bypassing HTTP entirely
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestAgentGraph:
    """Converted from scripts/test_agent_e2e.py."""

    @pytest.fixture(autouse=True)
    async def _init_opensearch(self) -> None:
        await init_opensearch()

    async def test_off_topic_query_is_rejected(self) -> None:
        result = await agent_graph.ainvoke({"query": "What's a good pizza topping?"})

        assert result["status"] == "off_topic"
        assert result["citations"] == []

    async def test_on_topic_query_produces_a_result(self) -> None:
        result = await agent_graph.ainvoke(
            {"query": "How does the transformer architecture work?"}
        )

        assert result["status"] in ("answered", "no_relevant_docs")
        assert isinstance(result.get("answer"), str)
        assert len(result["answer"]) > 0
        assert result["retrieval_attempts"] >= 1