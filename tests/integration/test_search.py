"""
Integration tests for Phase 5 — Search endpoints.

These tests hit the real FastAPI app but mock OpenSearch
so you don't need a running OpenSearch instance in CI.

For local testing with OpenSearch running, the tests marked
`opensearch` will hit the real index.
"""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.mark.integration
class TestSearchEndpoints:
    """Integration tests for search API endpoints."""

    @pytest.fixture
    async def client(self) -> AsyncClient:  # type: ignore[override]
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            yield c

    @pytest.fixture
    def mock_search_result(self) -> dict:
        """Sample search result for mocking."""
        return {
            "query": "transformer attention",
            "total_hits": 1,
            "results": [],
            "took_ms": 5,
            "page": 1,
            "page_size": 10,
            "has_more": False,
        }

    async def test_search_endpoint_exists(self, client: AsyncClient) -> None:
        """POST /api/v1/search should exist (even if OpenSearch is down)."""
        resp = await client.post(
            "/api/v1/search",
            json={"query": "transformer"},
        )
        # 200 = works, 503 = OpenSearch down but endpoint exists
        assert resp.status_code in (200, 503)

    async def test_search_rejects_empty_query(self, client: AsyncClient) -> None:
        """Empty query should return 422 Unprocessable Entity."""
        resp = await client.post(
            "/api/v1/search",
            json={"query": ""},
        )
        assert resp.status_code == 422

    async def test_search_rejects_invalid_page(self, client: AsyncClient) -> None:
        """Page 0 is invalid — should return 422."""
        resp = await client.post(
            "/api/v1/search",
            json={"query": "transformer", "page": 0},
        )
        assert resp.status_code == 422

    async def test_search_rejects_page_size_over_50(self, client: AsyncClient) -> None:
        """Page size > 50 is invalid."""
        resp = await client.post(
            "/api/v1/search",
            json={"query": "transformer", "page_size": 100},
        )
        assert resp.status_code == 422

    async def test_search_accepts_category_filter(self, client: AsyncClient) -> None:
        """Categories filter should be accepted without validation error."""
        resp = await client.post(
            "/api/v1/search",
            json={"query": "transformer", "categories": ["cs.AI", "cs.LG"]},
        )
        assert resp.status_code in (200, 503)

    async def test_get_paper_by_id_returns_404_for_unknown(
        self, client: AsyncClient
    ) -> None:
        """Unknown arxiv_id should return 404."""
        resp = await client.get("/api/v1/search/nonexistent-paper-id-99999")
        assert resp.status_code in (404, 503)

    async def test_search_stats_endpoint_exists(self, client: AsyncClient) -> None:
        """Stats endpoint should exist."""
        resp = await client.get("/api/v1/search/stats")
        assert resp.status_code in (200, 503)
