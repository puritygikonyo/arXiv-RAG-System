"""
Semantic query cache backed by Upstash Redis.

Upstash's free-tier REST API is plain key-value, not a vector index, so
similarity comparison happens in-process: we keep a small index (a Redis
set) of cache-entry keys, fetch them all, and compute cosine similarity
against the incoming query's embedding with numpy. Fine at this project's
scale (dozens-to-low-hundreds of cached queries) — would need a real
vector store (or Upstash Vector) if the cache grew into the thousands.

Cache entries are stored as JSON:
    {
        "query": "<original query text>",
        "embedding": [0.123, -0.045, ...],
        "answer": "<generated answer>",
    }

Usage:
    from src.services.cache.semantic_cache import get_cached_answer, set_cached_answer

    cached = await get_cached_answer("What is attention in transformers?")
    if cached is not None:
        return cached["answer"]

    # ... run the full agent ...
    await set_cached_answer(query, answer)
"""

import asyncio
import json
import uuid

import numpy as np

from src.config import get_settings
from src.logger import get_logger
from src.services.cache.redis_client import get_redis_client
from src.services.embeddings.jina import embed_for_similarity
from src.services.rate_limit import jina_semaphore


logger = get_logger(__name__)

# Redis key for the set holding every cache entry's key (the index).
CACHE_INDEX_KEY = "cache:index"
CACHE_ENTRY_PREFIX = "cache:entry:"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


async def get_cached_answer(query: str) -> dict | None:
    """
    Check the semantic cache for a query similar enough to an existing
    cached entry. Returns the cached entry dict (query/embedding/answer)
    on a hit, or None on a miss.
    """
    import time  # temporary, for diagnosing latency — remove after profiling

    settings = get_settings()
    redis = get_redis_client()

    t0 = time.monotonic()
    try:
        async with jina_semaphore:
            query_vector = await asyncio.to_thread(embed_for_similarity, query)
    except Exception as exc:
        logger.error("cache_embed_failed", error=str(exc), query=query)
        return None
    t1 = time.monotonic()
    logger.info("cache_timing_embed", seconds=round(t1 - t0, 3))

    try:
        entry_keys = await redis.smembers(CACHE_INDEX_KEY)
    except Exception as exc:
        logger.error("cache_index_read_failed", error=str(exc))
        return None
    t2 = time.monotonic()
    logger.info("cache_timing_smembers", seconds=round(t2 - t1, 3))

    if not entry_keys:
        logger.info("cache_miss", query=query[:100], reason="empty_index")
        return None

    entry_keys = list(entry_keys)

    try:
        raw_entries = await redis.mget(*entry_keys)
    except Exception as exc:
        logger.error("cache_entries_read_failed", error=str(exc))
        return None
    t3 = time.monotonic()
    logger.info("cache_timing_mget", seconds=round(t3 - t2, 3))

    

    best_entry = None
    best_score = 0.0
    stale_keys = []

    for key, raw in zip(entry_keys, raw_entries):
        if raw is None:
            # entry expired via TTL but index still references it —
            # queue it up for removal from the index below.
            stale_keys.append(key)
            continue
        try:
            entry = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue

        score = _cosine_similarity(query_vector, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if stale_keys:
        try:
            await redis.srem(CACHE_INDEX_KEY, *stale_keys)
            logger.info("cache_stale_keys_pruned", count=len(stale_keys))
        except Exception as exc:
            logger.error("cache_stale_prune_failed", error=str(exc))
            # non-fatal — a stray key just gets retried next lookup

    if best_entry is not None and best_score >= settings.cache_similarity_threshold:
        logger.info(
            "cache_hit",
            query=query[:100],
            matched_query=best_entry["query"][:100],
            similarity=round(best_score, 4),
        )
        return best_entry

    logger.info(
        "cache_miss",
        query=query[:100],
        best_similarity=round(best_score, 4),
        threshold=settings.cache_similarity_threshold,
    )
    return None


async def set_cached_answer(query: str, answer: str, citations: list[str] | None = None) -> None:
    """
    Store a new query/answer pair in the semantic cache. Re-embeds the
    query (cheap, and keeps this function independent of get_cached_answer's
    internals) and writes both the entry and its index membership.
    """
    settings = get_settings()
    redis = get_redis_client()

    try:
        query_vector = await asyncio.to_thread(embed_for_similarity, query)
    except Exception as exc:
        logger.error("cache_write_embed_failed", error=str(exc), query=query)
        return  # don't cache what we can't embed

    entry_key = f"{CACHE_ENTRY_PREFIX}{uuid.uuid4().hex}"
    entry = {
        "query": query,
        "embedding": query_vector,
        "answer": answer,
        "citations": citations or [],
    }

    try:
        await redis.set(entry_key, json.dumps(entry), ex=settings.cache_ttl_seconds)
        await redis.sadd(CACHE_INDEX_KEY, entry_key)
    except Exception as exc:
        logger.error("cache_write_failed", error=str(exc), query=query)
        return

    logger.info("cache_write_complete", query=query[:100], key=entry_key)

async def invalidate_cache() -> int:
    """
    Wipe the entire semantic cache.

    Called after new papers are ingested (Airflow DAG hook) — an answer
    cached before new papers existed may now be stale or incomplete
    relative to what retrieval would return today. Fails closed on error:
    if we can't confirm the wipe succeeded, we don't want to claim it did.
    """
    redis = get_redis_client()

    try:
        entry_keys = await redis.smembers(CACHE_INDEX_KEY)
    except Exception as exc:
        logger.error("cache_invalidate_read_failed", error=str(exc))
        raise

    if not entry_keys:
        logger.info("cache_invalidate_complete", count=0)
        return 0

    entry_keys = list(entry_keys)

    try:
        await redis.delete(*entry_keys)
        await redis.delete(CACHE_INDEX_KEY)
    except Exception as exc:
        logger.error("cache_invalidate_write_failed", error=str(exc), count=len(entry_keys))
        raise

    logger.info("cache_invalidate_complete", count=len(entry_keys))
    return len(entry_keys)