"""
Retrieval node: takes state["search_query"], embeds it via Jina, and
searches the arxiv_chunks index via vector search (hybrid.py's
search_chunks_by_vector). Both the embedding call and the OpenSearch
call are synchronous functions, so we run them in a thread to avoid
blocking the event loop while other requests are being served.
"""

import asyncio

from src.config import get_settings
from src.logger import get_logger
from src.services.agents.state import AgentState, ChunkResult
from src.services.embeddings.jina import embed_query
from src.services.search.hybrid import search_chunks_by_vector

logger = get_logger(__name__)
settings = get_settings()

# how many chunks to pull back per retrieval pass
TOP_K = 5


async def retriever_node(state: AgentState) -> dict:
    """Embed the current search query and fetch the most relevant chunks."""
    query = state["search_query"]
    attempt = state.get("retrieval_attempts", 0)

    # ── Step 1: embed the query ──────────────────────────────────────────
    try:
        query_vector = await asyncio.to_thread(embed_query, query)
    except Exception as exc:
        logger.error("retriever_embed_failed", error=str(exc), query=query)
        return {
            "chunks": [],
            "retrieval_attempts": attempt + 1,
            "status": "no_relevant_docs",
        }

    # ── Step 2: vector search the chunks index ──────────────────────────
    try:
        raw_chunks = await asyncio.to_thread(
            search_chunks_by_vector,
            query_vector,
            top_k=TOP_K,
        )
    except Exception as exc:
        logger.error("retriever_search_failed", error=str(exc), query=query)
        return {
            "chunks": [],
            "retrieval_attempts": attempt + 1,
            "status": "no_relevant_docs",
        }

    # ── Step 3: map hybrid.py's ChunkResult dataclass onto our AgentState
    # ChunkResult TypedDict. Field names differ on purpose — the agent's
    # internal state shouldn't be coupled to how the search layer names
    # things (arxiv_id vs paper_id, vector_score vs score, etc).
    chunks: list[ChunkResult] = [
        {
            "chunk_id": c.chunk_id,
            "paper_id": c.arxiv_id,
            "paper_title": c.title,
            "text": c.text,
            "score": c.vector_score,
            "relevance": 0.0,   # the grader node fills this in next
        }
        for c in raw_chunks
    ]

    logger.info(
        "retriever_complete",
        query=query[:100],
        attempt=attempt,
        chunks_found=len(chunks),
    )

    return {
        "chunks": chunks,
        "retrieval_attempts": attempt + 1,
    }