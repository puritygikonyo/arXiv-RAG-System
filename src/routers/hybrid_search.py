"""
Hybrid search router — POST /api/v1/hybrid-search

This endpoint:
  1. Receives a query string
  2. Embeds it using Jina AI
  3. Runs BM25 + vector search in parallel
  4. Merges results with RRF
  5. Returns ranked papers
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.logger import get_logger
from src.schemas.search import (
    HighlightSchema,
    PaperResultSchema,
    SearchResponseSchema,
)
from src.services.embeddings.jina import embed_query
from src.services.embeddings.vector_indexer import (
    ensure_chunks_index_exists,
    index_paper_with_embeddings,
)
from src.services.search.hybrid import search_chunks_by_vector, search_hybrid_papers

router = APIRouter(tags=["hybrid-search"])
logger = get_logger(__name__)


# ── Request / Response schemas ─────────────────────────────────────────────────

class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    categories: list[str] | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=50)


class ChunkResultSchema(BaseModel):
    chunk_id: str
    arxiv_id: str
    text: str
    title: str
    chunk_index: int
    total_chunks: int
    primary_category: str
    vector_score: float


class EmbedAndIndexRequest(BaseModel):
    """Request to embed and index a paper's chunks."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = []
    primary_category: str = ""
    categories: list[str] = []
    published_at: str | None = None
    pdf_url: str = ""


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/hybrid-search",
    response_model=SearchResponseSchema,
    summary="Hybrid search (BM25 + vector)",
    description=(
        "Search papers using hybrid BM25 + semantic vector search. "
        "Better recall than keyword-only search — finds papers "
        "with matching MEANING even if exact words differ. "
        "Requires papers to be indexed with embeddings first."
    ),
)
async def hybrid_search(request: HybridSearchRequest) -> SearchResponseSchema:
    """
    Hybrid search combining keyword (BM25) and semantic (vector) search.

    Step 1: Embed the query using Jina AI
    Step 2: Run BM25 search + vector search
    Step 3: Merge with RRF
    Step 4: Return ranked results
    """
    logger.info("hybrid_search_request", query=request.query, page=request.page)

    # Step 1: embed the query
    try:
        query_vector = embed_query(request.query)
    except ValueError as e:
        # JINA_API_KEY not configured
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error("query_embedding_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}",
        )

    # Step 2 + 3: hybrid search with RRF
    try:
        result = search_hybrid_papers(
            query=request.query,
            query_vector=query_vector,
            categories=request.categories,
            page=request.page,
            page_size=request.page_size,
        )
    except Exception as e:
        logger.error("hybrid_search_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search failed: {str(e)}",
        )

    paper_results = [
        PaperResultSchema(
            arxiv_id=r.arxiv_id,
            title=r.title,
            abstract=r.abstract,
            authors=r.authors,
            primary_category=r.primary_category,
            categories=r.categories,
            published_at=r.published_at,
            pdf_url=r.pdf_url,
            score=r.score,
            highlights=HighlightSchema(
                title=r.highlights.get("title", []),
                abstract=r.highlights.get("abstract", []),
            ),
        )
        for r in result.results
    ]

    has_more = (request.page * request.page_size) < result.total_hits

    return SearchResponseSchema(
        query=result.query,
        total_hits=result.total_hits,
        results=paper_results,
        took_ms=result.took_ms,
        page=result.page,
        page_size=result.page_size,
        has_more=has_more,
    )


@router.post(
    "/hybrid-search/chunks",
    response_model=list[ChunkResultSchema],
    summary="Vector search on chunks",
    description=(
        "Search paper chunks by semantic similarity. "
        "Returns the most relevant SECTIONS of papers, not whole papers. "
        "Used internally by the LangGraph agent in Phase 7."
    ),
)
async def search_chunks(request: HybridSearchRequest) -> list[ChunkResultSchema]:
    """
    Search chunks by vector similarity.
    Returns specific sections of papers most relevant to the query.
    """
    try:
        query_vector = embed_query(request.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    chunks = search_chunks_by_vector(
        query_vector=query_vector,
        categories=request.categories,
        top_k=request.page_size,
    )

    return [
        ChunkResultSchema(
            chunk_id=c.chunk_id,
            arxiv_id=c.arxiv_id,
            text=c.text,
            title=c.title,
            chunk_index=c.chunk_index,
            total_chunks=c.total_chunks,
            primary_category=c.primary_category,
            vector_score=c.vector_score,
        )
        for c in chunks
    ]


@router.post(
    "/hybrid-search/embed-and-index",
    summary="Embed and index a paper",
    description="Chunk a paper, generate embeddings, and index into OpenSearch.",
)
async def embed_and_index(request: EmbedAndIndexRequest) -> dict:
    """
    Full embedding pipeline for a single paper.
    Chunks the paper, generates embeddings, stores in OpenSearch.
    """
    ensure_chunks_index_exists()

    try:
        result = index_paper_with_embeddings(request.model_dump())
        return {
            "success": True,
            "arxiv_id": request.arxiv_id,
            **result,
        }
    except Exception as e:
        logger.error("embed_and_index_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
