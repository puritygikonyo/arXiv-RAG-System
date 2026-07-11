"""
BM25 Search Service — finds papers matching a user's query.

HOW BM25 SEARCH WORKS (step by step):

  User types: "attention mechanism in transformers"

  Step 1 — Tokenisation
    OpenSearch splits query into tokens: ["attention", "mechanism", "transformers"]
    Stop words removed: "in" is removed

  Step 2 — IDF (Inverse Document Frequency)
    Rare words get higher weight.
    If "transformers" appears in 10 papers → high IDF score (rare = important)
    If "the" appears in 10,000 papers → low IDF score (common = less important)

  Step 3 — TF (Term Frequency)
    Words that appear many times in a paper get higher weight.
    But there's diminishing returns — appearing 10x isn't 10x better than 5x.
    This prevents papers from gaming search by repeating keywords.

  Step 4 — Field boosting
    Title match counts 3x more than abstract match (we set this in mappings.py)
    Abstract match counts 2x more than author match

  Step 5 — Ranking
    BM25 combines IDF + TF + field boost into a single relevance score.
    Papers are returned sorted by score, highest first.

QUERY TYPES WE USE:

  multi_match:
    Searches across multiple fields (title, abstract, authors) simultaneously.
    The best field score wins. Good for general search.

  filter:
    Exact match — doesn't affect ranking score, just includes/excludes.
    Used for category filtering: "only show cs.AI papers"

  range:
    Date range queries: "papers published after 2024-01-01"
"""

from dataclasses import dataclass
from datetime import datetime

from src.config import get_settings
from src.logger import get_logger
from src.services.search.client import get_opensearch_client

logger = get_logger(__name__)
settings = get_settings()


# -----------------------------------------------------------------------------
# Result data structures
# Using dataclasses — lightweight, typed, no ORM overhead
# -----------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single paper returned from search."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published_at: str | None
    pdf_url: str
    score: float           # BM25 relevance score — higher = more relevant
    highlights: dict       # Which parts of the text matched the query


@dataclass
class SearchResponse:
    """The full response from a search query."""
    query: str
    total_hits: int        # total matching papers (before pagination)
    results: list[SearchResult]
    took_ms: int           # how long OpenSearch took to execute the query
    page: int
    page_size: int


# -----------------------------------------------------------------------------
# Main search function
# -----------------------------------------------------------------------------

def search_papers(
    query: str,
    *,
    categories: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    page: int = 1,
    page_size: int = 10,
    min_score: float | None = None,
) -> SearchResponse:
    """
    Search papers using BM25 keyword search.

    Args:
        query:       The search text e.g. "attention mechanism transformer"
        categories:  Filter to specific arXiv categories e.g. ["cs.AI", "cs.LG"]
        date_from:   Only return papers published after this date
        date_to:     Only return papers published before this date
        page:        Page number for pagination (starts at 1)
        page_size:   Number of results per page (max 50)
        min_score:   Minimum relevance score threshold (filters weak matches)

    Returns:
        SearchResponse with ranked list of matching papers

    PAGINATION EXPLAINED:
        page=1, page_size=10 → results 1-10   (from=0,  size=10)
        page=2, page_size=10 → results 11-20  (from=10, size=10)
        page=3, page_size=10 → results 21-30  (from=20, size=10)
        Formula: from = (page - 1) * page_size
    """
    client = get_opensearch_client()

    # Clamp page_size to reasonable limits
    page_size = min(page_size, 50)
    from_offset = (page - 1) * page_size

    # ------------------------------------------------------------------
    # Build the OpenSearch query
    # ------------------------------------------------------------------
    # We use a `bool` query — the most flexible query type in OpenSearch.
    # A bool query has four clauses:
    #   must:    conditions that MUST match (affects score)
    #   should:  conditions that SHOULD match (boosts score if they do)
    #   filter:  conditions that MUST match (does NOT affect score)
    #   must_not: conditions that MUST NOT match

    must_clauses = []
    filter_clauses = []

    # ── Main search query ──────────────────────────────────────────────
    if query.strip():
        must_clauses.append({
            "multi_match": {
                # Search these fields simultaneously
                # The ^ notation is field-level boosting:
                # title^3 means title matches are 3x more valuable
                "query": query,
                "fields": ["title^3", "abstract^2", "authors", "full_text"],
                "type": "best_fields",     # use the best matching field's score
                "fuzziness": "AUTO",       # handles typos: "transfomer" → "transformer"
                "minimum_should_match": "70%",  # at least 70% of words must match
                                                # prevents returning irrelevant results
            }
        })
    else:
        # Empty query → return all papers sorted by date
        must_clauses.append({"match_all": {}})

    # ── Category filter ────────────────────────────────────────────────
    if categories:
        filter_clauses.append({
            "terms": {
                "categories": categories   # paper must be in at least one category
            }
        })

    # ── Date range filter ──────────────────────────────────────────────
    if date_from or date_to:
        date_range: dict = {}
        if date_from:
            date_range["gte"] = date_from.isoformat()  # gte = greater than or equal
        if date_to:
            date_range["lte"] = date_to.isoformat()    # lte = less than or equal

        filter_clauses.append({
            "range": {"published_at": date_range}
        })

    # ── Assemble the full query ────────────────────────────────────────
    opensearch_query: dict = {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
            }
        },

        # Highlighting: OpenSearch marks which parts of the text matched
        # This is how search engines show you the bold snippets in results
        "highlight": {
            "fields": {
                "title": {
                    "number_of_fragments": 1,
                    "fragment_size": 200,
                    "pre_tags": ["<mark>"],    # wrap matched text in <mark>
                    "post_tags": ["</mark>"],
                },
                "abstract": {
                    "number_of_fragments": 2,
                    "fragment_size": 300,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
            }
        },

        # Minimum score filter — removes low-quality matches
        "min_score": min_score or settings.search_min_score,

        # Pagination
        "from": from_offset,
        "size": page_size,

        # Sort: primary sort by score, secondary sort by date (newest first)
        "sort": [
            {"_score": {"order": "desc"}},
            {"published_at": {"order": "desc"}},
        ],
    }

    # ------------------------------------------------------------------
    # Execute the query
    # ------------------------------------------------------------------
    logger.info(
        "search_executing",
        query=query,
        categories=categories,
        page=page,
        page_size=page_size,
    )

    try:
        response = client.search(
            index=settings.opensearch_papers_index,
            body=opensearch_query,
        )
    except Exception as e:
        logger.error("search_failed", query=query, error=str(e))
        raise

    # ------------------------------------------------------------------
    # Parse the response into our clean data structures
    # ------------------------------------------------------------------
    hits = response["hits"]
    total_hits = hits["total"]["value"]
    took_ms = response["took"]

    results = []
    for hit in hits["hits"]:
        source = hit["_source"]
        highlight = hit.get("highlight", {})

        results.append(SearchResult(
            arxiv_id=source["arxiv_id"],
            title=source["title"],
            abstract=source["abstract"],
            authors=source.get("authors", []),
            primary_category=source.get("primary_category", ""),
            categories=source.get("categories", []),
            published_at=source.get("published_at"),
            pdf_url=source.get("pdf_url", ""),
            score=round(hit["_score"], 4),
            highlights={
                "title": highlight.get("title", []),
                "abstract": highlight.get("abstract", []),
            },
        ))

    logger.info(
        "search_complete",
        query=query,
        total_hits=total_hits,
        returned=len(results),
        took_ms=took_ms,
    )

    return SearchResponse(
        query=query,
        total_hits=total_hits,
        results=results,
        took_ms=took_ms,
        page=page,
        page_size=page_size,
    )


def get_paper_by_id(arxiv_id: str) -> SearchResult | None:
    """
    Retrieve a specific paper from OpenSearch by its arXiv ID.

    Args:
        arxiv_id: e.g. "2401.12345"

    Returns:
        SearchResult if found, None if not in index
    """
    client = get_opensearch_client()

    try:
        response = client.get(
            index=settings.opensearch_papers_index,
            id=arxiv_id,
        )
    except Exception:
        return None

    if not response["found"]:
        return None

    source = response["_source"]
    return SearchResult(
        arxiv_id=source["arxiv_id"],
        title=source["title"],
        abstract=source["abstract"],
        authors=source.get("authors", []),
        primary_category=source.get("primary_category", ""),
        categories=source.get("categories", []),
        published_at=source.get("published_at"),
        pdf_url=source.get("pdf_url", ""),
        score=1.0,
        highlights={},
    )
