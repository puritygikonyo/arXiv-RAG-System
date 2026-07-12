"""
Hybrid Search Service — combines BM25 keyword search with vector semantic search.

THE CORE IDEA:

  BM25 finds papers with MATCHING WORDS.
  Vector search finds papers with MATCHING MEANING.
  Hybrid search gets the best of both.

  Example where BM25 fails but vector search wins:
    Query: "gradient vanishing problem"
    Paper: "difficulty training deep networks due to exploding gradients"
    → BM25 score: 0 (no word overlap)
    → Vector similarity: 0.87 (same concept, different words)

  Example where vector search fails but BM25 wins:
    Query: "BERT-base-uncased"
    Paper: "We use BERT-base-uncased for our experiments"
    → BM25 score: high (exact model name match)
    → Vector similarity: mediocre (model names are hard to embed well)

  Hybrid search catches BOTH cases.

HOW RRF WORKS (Reciprocal Rank Fusion):

  Instead of combining raw scores (which have different scales),
  RRF combines RANK POSITIONS.

  Paper A ranks #1 in BM25,  #3 in vector → RRF score = 1/(60+1) + 1/(60+3)
  Paper B ranks #5 in BM25,  #1 in vector → RRF score = 1/(60+5) + 1/(60+1)
  Paper C ranks #2 in BM25,  #2 in vector → RRF score = 1/(60+2) + 1/(60+2)

  The 60 is a constant (k=60 is the standard) that prevents top-ranked
  documents from dominating too much.

  Paper C (consistently ranked #2 in both) usually wins — this is the
  "consensus" signal RRF rewards.

TWO SEARCH STRATEGIES:

  1. search_hybrid_chunks():
     Searches the CHUNKS index with vector search.
     Best for finding the specific SECTION of a paper that answers a question.
     Used by the LangGraph agent in Phase 7.

  2. search_hybrid_papers():
     Searches the PAPERS index combining BM25 + vector at paper level.
     Best for finding PAPERS to recommend.
     Used by the /hybrid-search endpoint.
"""

from dataclasses import dataclass

from src.config import get_settings
from src.logger import get_logger
from src.services.search.bm25 import SearchResponse, SearchResult, search_papers
from src.services.search.client import get_opensearch_client

logger = get_logger(__name__)
settings = get_settings()

# RRF constant — standard value is 60
RRF_K = 60


@dataclass
class ChunkResult:
    """A single chunk returned from vector search."""
    chunk_id: str
    arxiv_id: str
    text: str
    title: str
    chunk_index: int
    total_chunks: int
    primary_category: str
    vector_score: float


def search_hybrid_papers(
    query: str,
    query_vector: list[float],
    *,
    categories: list[str] | None = None,
    page: int = 1,
    page_size: int = 10,
) -> SearchResponse:
    """
    Hybrid search combining BM25 and vector search at the PAPER level.

    Uses Reciprocal Rank Fusion to merge results from both systems.

    Args:
        query:        the user's search text
        query_vector: the embedded vector of the query (from Jina)
        categories:   optional category filter
        page:         page number
        page_size:    results per page

    Returns:
        SearchResponse with papers ranked by combined BM25 + vector score
    """
    logger.info("hybrid_search_papers", query=query[:100], page=page)

    # ── Step 1: BM25 keyword search ──────────────────────────────────────────
    bm25_response = search_papers(
        query=query,
        categories=categories,
        page=1,
        page_size=50,   # get more results than needed so RRF has enough to work with
        min_score=0.0,  # don't filter by score — RRF handles relevance
    )

    # Build BM25 rank map: {arxiv_id: rank_position}
    bm25_ranks: dict[str, int] = {
        result.arxiv_id: rank
        for rank, result in enumerate(bm25_response.results, start=1)
    }

    # ── Step 2: Vector (k-NN) search ─────────────────────────────────────────
    vector_results = _vector_search_papers(
        query_vector=query_vector,
        categories=categories,
        top_k=50,
    )

    # Build vector rank map: {arxiv_id: rank_position}
    vector_ranks: dict[str, int] = {
        result["arxiv_id"]: rank
        for rank, result in enumerate(vector_results, start=1)
    }

    # ── Step 3: RRF fusion ────────────────────────────────────────────────────
    # Collect all unique paper IDs from both result sets
    all_paper_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

    # Calculate RRF score for each paper
    # Formula: sum of 1/(k + rank) for each ranking list
    rrf_scores: dict[str, float] = {}
    for paper_id in all_paper_ids:
        bm25_rank = bm25_ranks.get(paper_id, 1000)   # not in BM25 → penalise
        vector_rank = vector_ranks.get(paper_id, 1000) # not in vector → penalise

        # Weight the two signals using the configured ratio
        bm25_weight = 1 - settings.hybrid_search_vector_weight
        vector_weight = settings.hybrid_search_vector_weight

        rrf_scores[paper_id] = (
            bm25_weight * (1 / (RRF_K + bm25_rank)) +
            vector_weight * (1 / (RRF_K + vector_rank))
        )

    # Sort by RRF score (highest first)
    ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    # ── Step 4: Paginate ─────────────────────────────────────────────────────
    from_offset = (page - 1) * page_size
    page_ids = ranked_ids[from_offset: from_offset + page_size]

    # ── Step 5: Build result objects ─────────────────────────────────────────
    # Look up full paper data from BM25 results (already fetched)
    bm25_by_id = {r.arxiv_id: r for r in bm25_response.results}

    # For papers only in vector results, fetch from OpenSearch
    vector_by_id = {r["arxiv_id"]: r for r in vector_results}

    results: list[SearchResult] = []
    for paper_id in page_ids:
        if paper_id in bm25_by_id:
            result = bm25_by_id[paper_id]
            # Replace score with RRF score
            result.score = round(rrf_scores[paper_id] * 1000, 4)
            results.append(result)
        elif paper_id in vector_by_id:
            vr = vector_by_id[paper_id]
            results.append(SearchResult(
                arxiv_id=vr["arxiv_id"],
                title=vr.get("title", ""),
                abstract=vr.get("abstract", ""),
                authors=vr.get("authors", []),
                primary_category=vr.get("primary_category", ""),
                categories=vr.get("categories", []),
                published_at=vr.get("published_at"),
                pdf_url=vr.get("pdf_url", ""),
                score=round(rrf_scores[paper_id] * 1000, 4),
                highlights={},
            ))

    logger.info(
        "hybrid_search_complete",
        query=query[:100],
        bm25_hits=len(bm25_ranks),
        vector_hits=len(vector_ranks),
        merged_total=len(all_paper_ids),
        returned=len(results),
    )

    return SearchResponse(
        query=query,
        total_hits=len(all_paper_ids),
        results=results,
        took_ms=0,
        page=page,
        page_size=page_size,
    )


def search_chunks_by_vector(
    query_vector: list[float],
    *,
    categories: list[str] | None = None,
    top_k: int = 5,
) -> list[ChunkResult]:
    """
    Search the CHUNKS index by vector similarity.

    This is what the LangGraph agent uses in Phase 7 —
    it finds the most semantically relevant SECTIONS of papers,
    not just which papers are relevant overall.

    Args:
        query_vector: embedded query from Jina
        categories:   optional category filter
        top_k:        number of chunks to return

    Returns:
        list of ChunkResult ordered by vector similarity
    """
    client = get_opensearch_client()

    query_body: dict = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        },
        "_source": [
            "chunk_id", "arxiv_id", "text", "title",
            "chunk_index", "total_chunks", "primary_category",
        ],
    }

    # Add category filter if requested
    if categories:
        query_body["query"] = {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": query_vector, "k": top_k}}}],
                "filter": [{"terms": {"primary_category": categories}}],
            }
        }

    try:
        response = client.search(
            index=settings.opensearch_chunks_index,
            body=query_body,
        )
    except Exception as e:
        logger.error("chunk_vector_search_failed", error=str(e))
        return []

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        results.append(ChunkResult(
            chunk_id=src["chunk_id"],
            arxiv_id=src["arxiv_id"],
            text=src["text"],
            title=src.get("title", ""),
            chunk_index=src.get("chunk_index", 0),
            total_chunks=src.get("total_chunks", 1),
            primary_category=src.get("primary_category", ""),
            vector_score=round(hit["_score"], 4),
        ))

    logger.info(
        "chunk_vector_search_complete",
        top_k=top_k,
        returned=len(results),
    )

    return results


def _vector_search_papers(
    query_vector: list[float],
    *,
    categories: list[str] | None = None,
    top_k: int = 50,
) -> list[dict]:
    """
    Internal: k-NN vector search on the PAPERS index.

    Returns raw dicts (not SearchResult) since we may not have
    all fields populated for papers only found via vector search.
    """
    client = get_opensearch_client()

    query_body: dict = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        },
        "_source": [
            "arxiv_id", "title", "abstract", "authors",
            "primary_category", "categories", "published_at", "pdf_url",
        ],
    }

    if categories:
        query_body["query"] = {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": query_vector, "k": top_k}}}],
                "filter": [{"terms": {"categories": categories}}],
            }
        }

    try:
        response = client.search(
            index=settings.opensearch_papers_index,
            body=query_body,
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        logger.warning("paper_vector_search_failed", error=str(e))
        return []
