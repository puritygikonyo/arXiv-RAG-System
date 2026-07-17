"""
Admin/operational endpoints — not part of the public API surface.
Guard these behind an API key before deploying anywhere public.
"""
from fastapi import APIRouter, Header, HTTPException

from src.config import get_settings
from src.logger import get_logger
from src.services.cache.semantic_cache import invalidate_cache
from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models import QueryLog


logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/cache/invalidate")
async def invalidate_semantic_cache(x_admin_key: str = Header(...)) -> dict:
    """
    Wipe the semantic cache. Called by the Airflow ingestion DAG after
    new papers are indexed, or manually during development.
    """
    settings = get_settings()
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    cleared = await invalidate_cache()
    logger.info("admin_cache_invalidate_triggered", cleared_count=cleared)
    return {"status": "ok", "entries_cleared": cleared}


@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Aggregate stats over query_logs: cache hit rate, avg latency by
    cache hit/miss, and the 10 most frequent queries. This is the data
    Langfuse's own dashboard doesn't aggregate natively — for per-node
    performance and individual request traces, use the Langfuse
    dashboard directly instead.
    """
    total = await db.scalar(select(func.count(QueryLog.id)))
    if not total:
        return {
            "total_queries": 0,
            "cache_hit_rate": 0.0,
            "avg_latency_ms": {"cache_hit": None, "cache_miss": None},
            "top_queries": [],
        }

    hits = await db.scalar(
        select(func.count(QueryLog.id)).where(QueryLog.cache_hit.is_(True))
    )
    cache_hit_rate = round(hits / total, 4)

    avg_hit_latency = await db.scalar(
        select(func.avg(QueryLog.latency_ms)).where(QueryLog.cache_hit.is_(True))
    )
    avg_miss_latency = await db.scalar(
        select(func.avg(QueryLog.latency_ms)).where(QueryLog.cache_hit.is_(False))
    )

    top_queries_result = await db.execute(
        select(QueryLog.query, func.count(QueryLog.id).label("count"))
        .group_by(QueryLog.query)
        .order_by(func.count(QueryLog.id).desc())
        .limit(10)
    )
    top_queries = [
        {"query": row.query, "count": row.count} for row in top_queries_result
    ]

    return {
        "total_queries": total,
        "cache_hit_rate": cache_hit_rate,
        "avg_latency_ms": {
            "cache_hit": round(avg_hit_latency, 1) if avg_hit_latency else None,
            "cache_miss": round(avg_miss_latency, 1) if avg_miss_latency else None,
        },
        "top_queries": top_queries,
    }