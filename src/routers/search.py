"""
Search router — exposes BM25 search via HTTP endpoints.

ENDPOINTS:
  POST /api/v1/search          → search papers by keyword
  GET  /api/v1/search/{id}     → get a specific paper by arXiv ID
  POST /api/v1/search/index    → manually index a paper (for testing)
  GET  /api/v1/search/stats    → index statistics

WHY POST FOR SEARCH?
  Normally GET is used for fetching data. But search requests can have
  complex bodies (filters, pagination, date ranges) that are awkward
  as query parameters. POST with a JSON body is cleaner for complex queries.

  Industry standard: Elasticsearch/OpenSearch use POST for search too.
"""

from fastapi import APIRouter, HTTPException, status

from src.logger import get_logger
from src.schemas.search import (
    IndexPaperRequest,
    IndexResponseSchema,
    PaperResultSchema,
    SearchRequest,
    SearchResponseSchema,
    HighlightSchema,
)
from src.services.search.bm25 import get_paper_by_id, search_papers
from src.services.search.indexer import (
    bulk_index_papers,
    get_index_stats,
    index_paper,
)

router = APIRouter(tags=["search"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# POST /api/v1/search — main search endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/search",
    response_model=SearchResponseSchema,
    summary="Search arXiv papers",
    description=(
        "Search papers using BM25 keyword search. "
        "Supports filtering by category and date range. "
        "Results are ranked by relevance score."
    ),
)
async def search(request: SearchRequest) -> SearchResponseSchema:
    """
    Search papers by keyword with optional filters.

    The query is matched against title (3x boost), abstract (2x boost),
    and authors (1x boost) using BM25 ranking.
    """
    logger.info(
        "search_request",
        query=request.query,
        categories=request.categories,
        page=request.page,
    )

    try:
        result = search_papers(
            query=request.query,
            categories=request.categories,
            date_from=request.date_from,
            date_to=request.date_to,
            page=request.page,
            page_size=request.page_size,
        )
    except Exception as e:
        logger.error("search_endpoint_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search service unavailable: {str(e)}",
        )

    # Convert dataclasses to Pydantic schemas for the response
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

    # Calculate if there are more pages
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


# ---------------------------------------------------------------------------
# GET /api/v1/search/{arxiv_id} — get specific paper
# ---------------------------------------------------------------------------
@router.get(
    "/search/{arxiv_id}",
    response_model=PaperResultSchema,
    summary="Get paper by arXiv ID",
)
async def get_paper(arxiv_id: str) -> PaperResultSchema:
    """
    Retrieve a specific paper from the search index by its arXiv ID.

    Returns 404 if the paper hasn't been indexed yet.
    """
    paper = get_paper_by_id(arxiv_id)

    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {arxiv_id} not found in search index",
        )

    return PaperResultSchema(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        primary_category=paper.primary_category,
        categories=paper.categories,
        published_at=paper.published_at,
        pdf_url=paper.pdf_url,
        score=paper.score,
        highlights=HighlightSchema(),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/search/index — manually index a paper
# ---------------------------------------------------------------------------
@router.post(
    "/search/index",
    response_model=IndexResponseSchema,
    summary="Index a paper manually",
    description="Manually index a single paper. Useful for testing.",
)
async def index_single_paper(request: IndexPaperRequest) -> IndexResponseSchema:
    """Index a single paper into OpenSearch."""
    try:
        index_paper(request.model_dump())
        return IndexResponseSchema(
            success=True,
            indexed=1,
            errors=0,
            message=f"Paper {request.arxiv_id} indexed successfully",
        )
    except Exception as e:
        logger.error("index_endpoint_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Indexing failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /api/v1/search/stats — index statistics
# ---------------------------------------------------------------------------
@router.get(
    "/search/stats",
    summary="Get search index statistics",
)
async def search_stats() -> dict:
    """Return statistics about the search index."""
    return get_index_stats()
