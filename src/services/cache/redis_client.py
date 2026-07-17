"""
Upstash Redis client wrapper.

Uses the REST API (not the TCP redis protocol) since Upstash's REST client
is HTTP-based — no persistent connection to manage, works cleanly in
serverless/async contexts. This mirrors how init_opensearch() is wired:
one cached client factory, reused everywhere.

Usage:
    from src.services.cache.redis_client import get_redis_client
    redis = get_redis_client()
    await redis.set("some_key", "some_value", ex=3600)
    value = await redis.get("some_key")
"""

from functools import lru_cache

from upstash_redis.asyncio import Redis

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_redis_client() -> Redis:
    """
    Return a cached Upstash Redis client instance.

    Building the client is just wiring config (URL + token) — no network
    call happens until the first command is awaited, so this is safe to
    call outside of an async context (e.g. at module import time).
    """
    settings = get_settings()

    if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
        logger.warning(
            "Upstash Redis credentials are missing or empty — "
            "cache operations will fail until UPSTASH_REDIS_REST_URL "
            "and UPSTASH_REDIS_REST_TOKEN are set in .env"
        )

    return Redis(
        url=settings.upstash_redis_rest_url,
        token=settings.upstash_redis_rest_token,
    )


async def ping_redis() -> bool:
    """
    Health check for the /api/v1/health endpoint.

    Upstash's REST client doesn't have a native PING command exposed the
    same way TCP redis-py does, so we round-trip a throwaway SET/GET
    instead — cheap, and it proves both auth and network path work.
    """
    try:
        client = get_redis_client()
        await client.set("__health_check__", "ok", ex=10)
        result = await client.get("__health_check__")
        return result == "ok"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


def reset_redis_client() -> None:
    """
    Clear the cached client.

    There's no socket/pool to close (REST client, no persistent
    connection) — this just drops the cached instance so a future
    get_redis_client() call rebuilds it. Mainly useful in tests, or if
    credentials rotate at runtime.
    """
    get_redis_client.cache_clear()
    logger.info("Redis client cache cleared")