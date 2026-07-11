"""
Indexing service — puts papers into OpenSearch so they become searchable.

THE MENTAL MODEL:
  Postgres  = your filing cabinet (stores everything permanently)
  OpenSearch = your search index (optimised copy for finding things)

  When a new paper arrives:
    1. Save it to Postgres (source of truth)
    2. Index it in OpenSearch (makes it searchable)

  If OpenSearch data is lost, you can rebuild it from Postgres.
  If Postgres data is lost, you have a real problem.

KEY CONCEPTS:

  Document:
    In OpenSearch, each paper is a "document" — a JSON object.
    Just like a row in Postgres, but stored as JSON in a search index.

  Document ID:
    We use the arxiv_id as the OpenSearch document ID.
    This means indexing the same paper twice just updates it (upsert).
    No duplicates.

  Bulk indexing:
    Instead of indexing one paper at a time (slow), we send batches.
    OpenSearch processes 50-100 papers in one network round trip.
    This is 50-100x faster than one-by-one.
"""

from datetime import UTC, datetime

from src.config import get_settings
from src.logger import get_logger
from src.services.search.client import get_opensearch_client

logger = get_logger(__name__)
settings = get_settings()


def index_paper(paper: dict) -> None:
    """
    Index a single paper into OpenSearch.

    Args:
        paper: dict with keys: arxiv_id, title, abstract, authors,
               categories, primary_category, published_at, pdf_url

    Why index a single paper?
        Used when a new paper arrives from the Airflow pipeline.
        The DAG calls this right after saving to Postgres.
    """
    client = get_opensearch_client()

    # Build the document we'll store in OpenSearch
    # Note: full_text combines title + abstract for single-field BM25 search
    document = {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper.get("authors", []),
        "full_text": f"{paper['title']} {paper['abstract']}",
        "primary_category": paper.get("primary_category", ""),
        "categories": paper.get("categories", []),
        "published_at": paper.get("published_at"),
        "pdf_url": paper.get("pdf_url", ""),
        "ingested_at": datetime.now(UTC).isoformat(),
    }

    # Index the document
    # id=paper["arxiv_id"] means:
    #   - If this arxiv_id doesn't exist → create it
    #   - If it already exists → update it (upsert behaviour)
    response = client.index(
        index=settings.opensearch_papers_index,
        body=document,
        id=paper["arxiv_id"],
        refresh=True,   # make immediately searchable (slight performance cost)
                        # remove refresh=True in production for better throughput
    )

    logger.info(
        "paper_indexed",
        arxiv_id=paper["arxiv_id"],
        result=response["result"],  # "created" or "updated"
    )


def bulk_index_papers(papers: list[dict]) -> dict:
    """
    Index multiple papers in a single batch operation.

    Args:
        papers: list of paper dicts (same format as index_paper)

    Returns:
        dict with counts: {"indexed": N, "errors": N}

    WHY BULK?
        Indexing 100 papers one-by-one = 100 network round trips
        Bulk indexing 100 papers = 1 network round trip
        At scale this is the difference between minutes and seconds.

    HOW BULK FORMAT WORKS:
        OpenSearch bulk API expects alternating lines:
        Line 1: action line  → { "index": { "_id": "2401.12345" } }
        Line 2: document     → { "title": "...", "abstract": "..." }
        Line 1: action line  → { "index": { "_id": "2401.67890" } }
        Line 2: document     → { "title": "...", "abstract": "..." }
        ... and so on
    """
    if not papers:
        logger.warning("bulk_index_called_with_empty_list")
        return {"indexed": 0, "errors": 0}

    client = get_opensearch_client()

    # Build the bulk request body — alternating action + document pairs
    bulk_body = []
    for paper in papers:
        # Action line: tells OpenSearch what to do and which document ID to use
        action = {
            "index": {
                "_index": settings.opensearch_papers_index,
                "_id": paper["arxiv_id"],  # arxiv_id = document ID (auto-upsert)
            }
        }

        # Document line: the actual content to store
        document = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": paper.get("authors", []),
            "full_text": f"{paper['title']} {paper['abstract']}",
            "primary_category": paper.get("primary_category", ""),
            "categories": paper.get("categories", []),
            "published_at": paper.get("published_at"),
            "pdf_url": paper.get("pdf_url", ""),
            "ingested_at": datetime.now(UTC).isoformat(),
        }

        bulk_body.append(action)
        bulk_body.append(document)

    # Send everything in one request
    response = client.bulk(body=bulk_body, refresh=True)

    # Count successes and failures
    indexed = 0
    errors = 0

    for item in response["items"]:
        if item["index"]["status"] in (200, 201):
            indexed += 1
        else:
            errors += 1
            logger.error(
                "bulk_index_item_failed",
                arxiv_id=item["index"].get("_id"),
                error=item["index"].get("error"),
            )

    logger.info(
        "bulk_index_complete",
        total=len(papers),
        indexed=indexed,
        errors=errors,
    )

    return {"indexed": indexed, "errors": errors}


def delete_paper(arxiv_id: str) -> bool:
    """
    Remove a paper from the search index.

    Args:
        arxiv_id: the arXiv ID to remove e.g. "2401.12345"

    Returns:
        True if deleted, False if it didn't exist

    When would you use this?
        If a paper is retracted on arXiv, you'd want to remove it
        from search results too.
    """
    client = get_opensearch_client()

    try:
        response = client.delete(
            index=settings.opensearch_papers_index,
            id=arxiv_id,
        )
        deleted = response["result"] == "deleted"
        logger.info("paper_deleted_from_index", arxiv_id=arxiv_id, deleted=deleted)
        return deleted
    except Exception as e:
        logger.warning("paper_delete_failed", arxiv_id=arxiv_id, error=str(e))
        return False


def get_index_stats() -> dict:
    """
    Return stats about the papers index.
    Useful for the /health endpoint and monitoring.
    """
    client = get_opensearch_client()

    try:
        stats = client.indices.stats(index=settings.opensearch_papers_index)
        count_response = client.count(index=settings.opensearch_papers_index)

        return {
            "total_papers": count_response["count"],
            "index_size_bytes": stats["_all"]["total"]["store"]["size_in_bytes"],
            "index_name": settings.opensearch_papers_index,
        }
    except Exception as e:
        logger.error("index_stats_failed", error=str(e))
        return {"error": str(e)}
