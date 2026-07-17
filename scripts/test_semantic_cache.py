"""
Throwaway check for the semantic cache — confirms real Jina embeddings +
real Upstash Redis produce correct hit/miss behavior before this touches
the /ask flow. Delete this file afterwards (same pattern as the other
scripts/test_*.py debug files).

Usage:
    uv run python scripts/test_semantic_cache.py
"""

import asyncio

from src.services.cache.semantic_cache import get_cached_answer, set_cached_answer


async def main() -> None:
    original_query = "What is attention in transformers?"
    paraphrased_query = "What's the attention mechanism in a transformer model?"
    unrelated_query = "What's the capital of France?"
    test_answer = "This is a test answer written by test_semantic_cache.py."

    print(f"Writing to cache: {original_query!r}")
    await set_cached_answer(original_query, test_answer)

    print(f"\nLooking up paraphrase: {paraphrased_query!r}")
    hit = await get_cached_answer(paraphrased_query)
    if hit is not None:
        print(f"✅ HIT — matched query: {hit['query']!r}")
        assert hit["answer"] == test_answer, "answer mismatch on hit"
    else:
        print("❌ MISS — expected a hit here. Threshold may be too strict, "
              "or check the query wording is close enough.")

    print(f"\nLooking up unrelated query: {unrelated_query!r}")
    miss = await get_cached_answer(unrelated_query)
    if miss is None:
        print("✅ MISS — correct, unrelated query did not match.")
    else:
        print(f"❌ HIT — unexpected match on unrelated query: {miss['query']!r}")

    print("\nDone. Check the log lines above for cache_hit / cache_miss "
          "similarity scores to see how close the threshold call was.")


if __name__ == "__main__":
    asyncio.run(main())