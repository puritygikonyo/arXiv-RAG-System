"""
Throwaway connectivity check for Upstash Redis.

Run this once to confirm .env credentials actually work before we wire
caching into the /ask flow. Delete this file afterwards (same pattern as
the Phase 7 scripts/test_*.py debug files).

Usage:
    uv run python scripts/test_redis_connection.py
"""

import asyncio

from src.services.cache.redis_client import get_redis_client


async def main() -> None:
    redis = get_redis_client()

    print("Setting test key...")
    await redis.set("arxiv_rag:connectivity_check", "hello from phase 8", ex=60)

    print("Getting test key...")
    value = await redis.get("arxiv_rag:connectivity_check")
    print(f"Got back: {value!r}")

    assert value == "hello from phase 8", "Round-trip failed — value mismatch"

    print("Deleting test key...")
    await redis.delete("arxiv_rag:connectivity_check")

    print("\n✅ Upstash Redis round-trip confirmed working.")


if __name__ == "__main__":
    asyncio.run(main())