"""
Search request and response schemas.

WHY SCHEMAS?
  FastAPI uses these Pydantic models to:
  1. VALIDATE incoming requests automatically
     → If someone sends page=-1, FastAPI rejects it with a clear error
     → You never write manual validation code

  2. DOCUMENT the API automatically
     → FastAPI reads these models and generates the /docs Swagger UI
     → Anyone can see exactly what fields are required and what they mean

  3. SERIALISE outgoing responses
     → FastAPI converts your dataclasses/dicts to JSON automatically

  Think of schemas as contracts:
  "If you send me THIS, I will return THAT"
"""

from datetime import datetime

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Request schemas — what comes IN to the API
# -----------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """
    Body of POST /api/v1/search

    All fields except `query` are optional — they add filters on top of search.
    """

    query: str = Field(
        ...,                          # ... means required
        min_length=1,
        max_length=500,
        description="Search query text",
        examples=["attention mechanism transformer", "federated learning privacy"],
    )

    categories: list[str] | None = Field(
        default=None,
        description="Filter by arXiv categories e.g. ['cs.AI', 'cs.LG']",
        examples=[["cs.AI", "cs.LG"]],
    )

    date_from: datetime | None = Field(
        default=None,
        description="Only return papers published after this date",
        examples=["2024-01-01T00:00:00Z"],
    )

    date_to: datetime | None = Field(
        default=None,
        description="Only return papers published before this date",
        examples=["2024-12-31T23:59:59Z"],
    )

    page: int = Field(
        default=1,
        ge=1,               # ge = greater than or equal to 1
        description="Page number (starts at 1)",
    )

    page_size: int = Field(
        default=10,
        ge=1,
        le=50,              # le = less than or equal to 50
        description="Number of results per page (max 50)",
    )


# -----------------------------------------------------------------------------
# Response schemas — what goes OUT from the API
# -----------------------------------------------------------------------------

class HighlightSchema(BaseModel):
    """Highlighted text snippets showing WHERE the query matched."""
    title: list[str] = []       # highlighted title fragments
    abstract: list[str] = []    # highlighted abstract fragments


class PaperResultSchema(BaseModel):
    """A single paper in the search results."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published_at: str | None
    pdf_url: str
    score: float = Field(description="BM25 relevance score — higher = more relevant")
    highlights: HighlightSchema


class SearchResponseSchema(BaseModel):
    """
    Full response from POST /api/v1/search

    Includes metadata (total hits, timing) and the list of results.
    """
    query: str
    total_hits: int = Field(description="Total matching papers (before pagination)")
    results: list[PaperResultSchema]
    took_ms: int = Field(description="Time OpenSearch took to execute the query")
    page: int
    page_size: int
    has_more: bool = Field(description="Whether there are more pages after this one")


# -----------------------------------------------------------------------------
# Index request schema — for manually triggering indexing
# -----------------------------------------------------------------------------

class IndexPaperRequest(BaseModel):
    """Body of POST /api/v1/search/index — manually index a paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = []
    primary_category: str = ""
    categories: list[str] = []
    published_at: str | None = None
    pdf_url: str = ""


class IndexResponseSchema(BaseModel):
    """Response from indexing operations."""
    success: bool
    indexed: int
    errors: int
    message: str
