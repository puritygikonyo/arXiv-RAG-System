"""
Langfuse tracing client.

One cached client instance, same pattern as get_redis_client(): build once,
reuse everywhere. The SDK batches trace events and ships them to Langfuse
Cloud on a background thread, so calling .trace()/.span() in a hot request
path doesn't block on network I/O.

Usage:
    from src.services.monitoring.langfuse_client import get_langfuse_client
    langfuse = get_langfuse_client()
"""

from functools import lru_cache

from langfuse import Langfuse

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_langfuse_client() -> Langfuse:
    """
    Return a cached Langfuse client instance.

    Like get_redis_client(), building this is just wiring credentials —
    no network call happens until the first trace event is flushed.
    """
    settings = get_settings()

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse credentials are missing or empty — "
            "tracing calls will be no-ops until LANGFUSE_PUBLIC_KEY "
            "and LANGFUSE_SECRET_KEY are set in .env"
        )

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def flush_langfuse() -> None:
    """
    Force-send any queued trace events immediately.

    The SDK batches events and sends them on a background thread on its
    own schedule — fine for a long-running server, but call this
    explicitly on FastAPI shutdown so traces from the final requests
    before a restart aren't silently dropped when the process exits.
    """
    client = get_langfuse_client()
    client.flush()
    logger.info("Langfuse events flushed")