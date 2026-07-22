"""
OpenSearch client — manages the connection to OpenSearch.

PATTERN: Singleton
  We create ONE client at startup and reuse it everywhere.
  Creating a new client per request would be wasteful (like opening
  a new DB connection for every HTTP request).

HOW IT WORKS:
  1. App starts up (lifespan in main.py)
  2. init_opensearch() is called — creates client, verifies connection,
     creates index if it doesn't exist
  3. Every service calls get_opensearch_client() to get the shared client
  4. App shuts down — close_opensearch() is called

WHY A MODULE-LEVEL VARIABLE?
  FastAPI doesn't have built-in dependency injection for non-HTTP things
  like DB clients. A module-level singleton is the clean, standard pattern.

CONNECTION POOL SIZE (Phase 8 fix):
  Load testing under 10 concurrent users surfaced "Connection pool is
  full, discarding connection: localhost. Connection pool size: 1" —
  the underlying urllib3 HTTP connection pool was capped at 1, meaning
  only one OpenSearch request could be in flight across the entire app
  at once. Every concurrent request beyond that had to wait for the
  single connection to free up, or the pool discarded and recreated
  connections repeatedly. pool_maxsize explicitly raises this cap.

TLS CERT VERIFICATION (Phase 10 fix):
  verify_certs was previously hardcoded to False with a "local dev
  only — enable in production" comment that nobody had come back to
  yet. Now that we're pointing this at a real hosted cluster (Bonsai)
  over HTTPS with a real certificate, it's tied to opensearch_use_ssl
  instead: local dev (plain http, no SSL) doesn't need cert
  verification since there's no cert to verify, but any SSL connection
  gets real verification rather than blindly trusting whatever's on
  the other end of the connection.
"""

from opensearchpy import OpenSearch
from src.config import get_settings
from src.logger import get_logger
from src.services.search.mappings import PAPERS_INDEX_MAPPING

logger = get_logger(__name__)
settings = get_settings()

# Module-level singleton — None until init_opensearch() is called
_client: OpenSearch | None = None


def get_opensearch_client() -> OpenSearch:
    """
    Return the shared OpenSearch client.

    Raises RuntimeError if called before init_opensearch().
    This is intentional — fail loud and early rather than
    silently returning None and crashing later.
    """
    if _client is None:
        raise RuntimeError(
            "OpenSearch client not initialised. "
            "Call init_opensearch() during application startup."
        )
    return _client


async def init_opensearch() -> None:
    """
    Initialise the OpenSearch client and ensure the papers index exists.

    Called once during FastAPI lifespan startup.
    Creates the index with correct mappings if it doesn't already exist.
    """
    global _client

    logger.info("opensearch_connecting", url=settings.opensearch_url)

    # Create the client
    # Using synchronous client here for simplicity —
    # OpenSearch operations are fast enough that async isn't critical
    _client = OpenSearch(
        hosts=[{
            "host": settings.opensearch_host,
            "port": settings.opensearch_port,
        }],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_use_ssl,   # see TLS note above
        ssl_show_warn=settings.opensearch_use_ssl,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
        pool_maxsize=25,  # was implicitly 1 — raised to support concurrent requests
    )

    # Verify the connection works
    try:
        health = _client.cluster.health()
        logger.info(
            "opensearch_connected",
            status=health["status"],
            nodes=health["number_of_nodes"],
        )
    except Exception as e:
        logger.error("opensearch_connection_failed", error=str(e))
        raise

    # Create the papers index if it doesn't already exist
    await _ensure_index_exists()


async def _ensure_index_exists() -> None:
    """
    Create the papers index with correct mappings if it doesn't exist.

    Using `if not exists` pattern — safe to call multiple times.
    Won't overwrite existing data if index already exists.
    """
    client = get_opensearch_client()
    index_name = settings.opensearch_papers_index

    if client.indices.exists(index=index_name):
        logger.info("opensearch_index_exists", index=index_name)
        return

    logger.info("opensearch_index_creating", index=index_name)

    client.indices.create(
        index=index_name,
        body=PAPERS_INDEX_MAPPING,
    )

    logger.info("opensearch_index_created", index=index_name)


async def close_opensearch() -> None:
    """
    Close the OpenSearch client connection.
    Called during FastAPI lifespan shutdown.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("opensearch_disconnected")