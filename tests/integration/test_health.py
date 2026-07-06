"""
Integration tests for the health check endpoint.

These tests spin up the FastAPI app in-process using httpx.AsyncClient
and verify the /api/v1/health endpoint behaves correctly.

Marked as `integration` — requires no external services to pass,
but OpenSearch status will show as "unreachable" if not running.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for GET /api/v1/health."""

    @pytest.fixture
    async def client(self) -> AsyncClient:  # type: ignore[override]
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            yield c

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    async def test_health_response_has_required_fields(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "uptime_seconds" in data
        assert "services" in data

    async def test_health_services_include_all_expected(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        services = resp.json()["services"]
        assert "opensearch" in services
        assert "database" in services
        assert "redis" in services

    async def test_health_version_is_correct(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.json()["version"] == "0.1.0"

    async def test_health_environment_is_development(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.json()["environment"] == "development"

    async def test_health_uptime_is_positive(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.json()["uptime_seconds"] >= 0

    async def test_root_endpoint_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code == 200

    async def test_docs_endpoint_accessible(self, client: AsyncClient) -> None:
        resp = await client.get("/docs")
        assert resp.status_code == 200
