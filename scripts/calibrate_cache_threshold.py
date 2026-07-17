"""
Threshold calibration for the semantic cache.

Runs several real query pairs through embed_for_similarity + cosine
similarity and prints a table, so we pick cache_similarity_threshold
from actual data instead of one lucky/unlucky sample.

This does NOT touch Redis — pure embedding comparison, so it's cheap
to re-run while tuning. Delete once the threshold is locked in.

Usage:
    uv run python scripts/calibrate_cache_threshold.py
"""

import asyncio

from src.services.cache.semantic_cache import _cosine_similarity
from src.services.embeddings.jina import embed_for_similarity

# label, query_a, query_b, expected ("HIT" = should match, "MISS" = should not)
PAIRS = [
    # genuine paraphrases — SHOULD hit
    ("paraphrase", "What is attention in transformers?",
     "What's the attention mechanism in a transformer model?", "HIT"),
    ("paraphrase", "How does the transformer architecture work?",
     "Can you explain how transformers are architected?", "HIT"),
    ("paraphrase", "What is self-attention?",
     "Explain the concept of self-attention.", "HIT"),

    # related but genuinely different questions — SHOULD miss
    ("related-different", "What is attention in transformers?",
     "What is positional encoding in transformers?", "MISS"),
    ("related-different", "How does self-attention work?",
     "How does multi-head attention differ from self-attention?", "MISS"),
    ("related-different", "What is the transformer architecture?",
     "What are the limitations of the transformer architecture?", "MISS"),

    # unrelated — SHOULD miss
    ("unrelated", "What is attention in transformers?",
     "What's the capital of France?", "MISS"),
    ("unrelated", "How does self-attention work?",
     "What's a good recipe for pasta?", "MISS"),
]


async def main() -> None:
    results = []

    for label, query_a, query_b, expected in PAIRS:
        vec_a = await asyncio.to_thread(embed_for_similarity, query_a)
        vec_b = await asyncio.to_thread(embed_for_similarity, query_b)
        score = _cosine_similarity(vec_a, vec_b)
        results.append((label, query_a, query_b, expected, score))

    print(f"\n{'label':<20}{'expected':<10}{'score':<8}query pair")
    print("-" * 90)
    for label, query_a, query_b, expected, score in results:
        print(f"{label:<20}{expected:<10}{score:<8.4f}{query_a!r} vs {query_b!r}")

    hit_scores = [s for l, _, _, e, s in results if e == "HIT"]
    miss_scores = [s for l, _, _, e, s in results if e == "MISS"]

    print(f"\nHIT scores  (should cluster HIGH): min={min(hit_scores):.4f}  max={max(hit_scores):.4f}")
    print(f"MISS scores (should cluster LOW):  min={min(miss_scores):.4f}  max={max(miss_scores):.4f}")

    if min(hit_scores) > max(miss_scores):
        suggested = round((min(hit_scores) + max(miss_scores)) / 2, 2)
        print(f"\n✅ Clean separation. Midpoint suggests threshold ≈ {suggested}")
    else:
        print(
            "\n⚠️  Overlap between HIT and MISS scores — no single threshold "
            "perfectly separates these examples. Pick a value that leans "
            "toward fewer false positives (closer to the top of the MISS range)."
        )


if __name__ == "__main__":
    asyncio.run(main())