"""
Health check router.

GET /api/v1/health — returns status of all connected services.

This is the first endpoint you build at any company.
It's what your load balancer, Kubernetes readiness probe,
and on-call engineers hit first when something breaks.
"""

import time
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from src.config import get_settings
from src.logger import get_logger

router = APIRouter(tags=["health"])
logger = get_logger(__name__)
settings = get_settings()


# ── Response schemas ──────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    status: str          # "ok" | "degraded" | "unreachable"
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded" | "unhealthy"
    version: str
    environment: str
    uptime_seconds: float
    services: dict[str, ServiceStatus]


# Track app start time
_start_time = time.monotonic()


# ── Health check helpers ──────────────────────────────────────────────────────

async def check_opensearch() -> ServiceStatus:
    """Ping OpenSearch cluster health endpoint."""
    url = f"{settings.opensearch_url}/_cluster/health"
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return ServiceStatus(
                status="ok" if data.get("status") != "red" else "degraded",
                latency_ms=round(latency, 2),
                detail=f"cluster status: {data.get('status')}",
            )
        return ServiceStatus(
            status="degraded",
            latency_ms=round(latency, 2),
            detail=f"HTTP {resp.status_code}",
        )
    except Exception as e:
        return ServiceStatus(status="unreachable", detail=str(e))


async def check_database() -> ServiceStatus:
    """
    Check PostgreSQL connectivity.
    Phase 3 will replace this stub with a real DB ping.
    """
    # Stub — will be replaced in Phase 3 when we add SQLAlchemy
    if settings.database_url.startswith("postgresql+asyncpg://user:password"):
        return ServiceStatus(
            status="not_configured",
            detail="Set DATABASE_URL in .env (Phase 3)",
        )
    return ServiceStatus(status="ok", detail="connection check pending Phase 3")


async def check_redis() -> ServiceStatus:
    """
    Check Redis connectivity.
    Phase 8 will replace this stub with a real Redis ping.
    """
    if settings.redis_url == "redis://localhost:6379":
        return ServiceStatus(
            status="not_configured",
            detail="Set REDIS_URL in .env (Phase 8 — Upstash)",
        )
    return ServiceStatus(status="ok", detail="connection check pending Phase 8")


# ── Health endpoint ───────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description=(
        "Returns the status of all services. "
        "Used by load balancers and on-call engineers. "
        "Returns 200 even when degraded so the process stays in rotation."
    ),
)
async def health_check() -> HealthResponse:
    """Check health of all downstream services."""
    logger.debug("health_check_called")

    # Run all checks (will be concurrent in Phase 3+)
    opensearch_status = await check_opensearch()
    database_status = await check_database()
    redis_status = await check_redis()

    services: dict[str, Any] = {
        "opensearch": opensearch_status,
        "database": database_status,
        "redis": redis_status,
    }

    # Determine overall status
    statuses = [s.status for s in services.values()]
    if all(s in ("ok", "not_configured") for s in statuses):
        overall = "ok"
    elif "unreachable" in statuses:
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version="0.1.0",
        environment=settings.app_env,
        uptime_seconds=round(time.monotonic() - _start_time, 2),
        services=services,
    )
