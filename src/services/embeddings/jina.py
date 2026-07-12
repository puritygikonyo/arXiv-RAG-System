"""
Jina AI Embedding Client — converts text into vectors.

WHAT THIS DOES:
  Takes a piece of text and returns a list of 1024 numbers (a vector)
  that represents the MEANING of that text.

  "attention mechanism in transformers" → [0.21, -0.54, 0.88, ...]

WHY JINA AI?
  - Free tier: 1 million tokens included, no credit card needed
  - jina-embeddings-v3 is one of the best embedding models available
  - Simple REST API — just POST text, get back vectors
  - Sign up at: https://jina.ai → copy your API key to .env

HOW EMBEDDINGS ARE USED:
  1. When a paper is ingested → embed its title + abstract → store in OpenSearch
  2. When a user searches → embed their query → find similar vectors

BATCHING:
  Calling the API once per paper would be very slow (1 API call per paper).
  Instead we batch: send 50 papers in one API call → 50x faster.
  The Jina API accepts up to 2048 texts per request.

RETRY LOGIC:
  Network calls fail sometimes. We use tenacity to automatically retry
  with exponential backoff:
    Try 1: immediate
    Try 2: wait 1 second
    Try 3: wait 2 seconds
    Try 4: wait 4 seconds
  This handles temporary network blips without crashing.
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Jina AI API endpoint
JINA_API_URL = "https://api.jina.ai/v1/embeddings"

# Maximum texts per API call (Jina's limit is 2048, we use 50 to be safe)
BATCH_SIZE = 50


@retry(
    stop=stop_after_attempt(4),           # try up to 4 times
    wait=wait_exponential(min=1, max=10), # wait 1s, 2s, 4s between tries
    reraise=True,                         # if all retries fail, raise the error
)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Jina AI.

    Args:
        texts: list of strings to embed (max 50 per call)

    Returns:
        list of embedding vectors, one per input text
        each vector is a list of 1024 floats

    Raises:
        ValueError: if texts list is empty
        httpx.HTTPError: if Jina API call fails after all retries

    Example:
        vectors = embed_texts(["attention mechanism", "transformer model"])
        # vectors[0] is the embedding for "attention mechanism"
        # vectors[1] is the embedding for "transformer model"
        # len(vectors[0]) == 1024
    """
    if not texts:
        raise ValueError("texts list cannot be empty")

    if not settings.jina_api_key:
        raise ValueError(
            "JINA_API_KEY not set in .env — "
            "sign up free at https://jina.ai and add your key"
        )

    logger.info("embedding_texts", count=len(texts), model=settings.jina_embedding_model)

    response = httpx.post(
        JINA_API_URL,
        headers={
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.jina_embedding_model,
            "input": texts,
            # task type tells Jina HOW the embeddings will be used
            # "retrieval.passage" = for indexing documents
            # "retrieval.query" = for search queries
            # Using the right task type improves search quality
            "task": "retrieval.passage",
            "dimensions": settings.jina_embedding_dimensions,
        },
        timeout=60.0,  # Jina can be slow for large batches
    )

    response.raise_for_status()
    data = response.json()

    # Jina returns: {"data": [{"embedding": [...], "index": 0}, ...]}
    # Sort by index to maintain original order
    embeddings = sorted(data["data"], key=lambda x: x["index"])
    vectors = [item["embedding"] for item in embeddings]

    logger.info(
        "embedding_complete",
        count=len(vectors),
        dimensions=len(vectors[0]) if vectors else 0,
    )

    return vectors


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a single search query.

    Uses "retrieval.query" task type which is optimised for
    queries rather than documents — Jina uses different internal
    processing for queries vs passages.

    Args:
        query: the user's search query

    Returns:
        single embedding vector (list of 1024 floats)
    """
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not set in .env")

    logger.debug("embedding_query", query=query[:100])

    response = httpx.post(
        JINA_API_URL,
        headers={
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.jina_embedding_model,
            "input": [query],
            "task": "retrieval.query",   # ← different task type for queries
            "dimensions": settings.jina_embedding_dimensions,
        },
        timeout=30.0,
    )

    response.raise_for_status()
    data = response.json()

    vector = data["data"][0]["embedding"]
    logger.debug("query_embedding_complete", dimensions=len(vector))

    return vector


def embed_texts_batched(texts: list[str]) -> list[list[float]]:
    """
    Embed a large list of texts in batches to avoid API limits.

    For example, 200 papers → 4 batches of 50 → 4 API calls.
    Results are reassembled in the original order.

    Args:
        texts: any number of strings to embed

    Returns:
        list of embedding vectors in the same order as input
    """
    if not texts:
        return []

    all_vectors: list[list[float]] = []

    # Split into batches
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    logger.info(
        "embedding_batched",
        total_texts=len(texts),
        num_batches=len(batches),
        batch_size=BATCH_SIZE,
    )

    for batch_num, batch in enumerate(batches, start=1):
        logger.info(
            "embedding_batch",
            batch=batch_num,
            of=len(batches),
            size=len(batch),
        )
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)

    return all_vectors
