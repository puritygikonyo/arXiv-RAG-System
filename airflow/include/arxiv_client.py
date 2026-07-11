"""
arXiv API client — fetches papers from arXiv's public Atom feed API.

No API key needed; arXiv's API is free and open.
Docs: https://info.arxiv.org/help/api/user-manual.html
"""

import logging
from datetime import datetime

import feedparser
import requests

logger = logging.getLogger(__name__)

ARXIV_NAMESPACE_PREFIX = "{http://arxiv.org/schemas/atom}"


def fetch_papers_for_category(
    base_url: str,
    category: str,
    max_results: int = 100,
) -> list[dict]:
    """
    Query arXiv's API for the most recent papers in a given category.

    Returns a list of dicts, each shaped to match the `papers` table schema.
    """
    params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }

    logger.info("Fetching arXiv papers", extra={"category": category, "max_results": max_results})

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        try:
            papers.append(_parse_entry(entry))
        except Exception as e:
            # Skip malformed entries rather than failing the whole batch —
            # arXiv's feed occasionally has entries missing optional fields.
            logger.warning(
                "Skipping malformed arXiv entry",
                extra={"entry_id": getattr(entry, "id", "unknown"), "error": str(e)},
            )
            continue

    logger.info("Fetched papers", extra={"category": category, "count": len(papers)})
    return papers


def _parse_entry(entry) -> dict:
    """Convert a single feedparser entry into a dict matching the papers table."""
    # arXiv IDs come back as full URLs like:
    #   http://arxiv.org/abs/2301.12345v2
    # We want just "2301.12345" (drop the version suffix).
    raw_id = entry.id.split("/abs/")[-1]
    arxiv_id = raw_id.split("v")[0] if "v" in raw_id.split("/")[-1] else raw_id

    authors = [author.name for author in getattr(entry, "authors", [])]

    categories = [tag["term"] for tag in getattr(entry, "tags", [])]
    primary_category = getattr(entry, "arxiv_primary_category", {}).get(
        "term", categories[0] if categories else "unknown"
    )

    pdf_url = ""
    for link in getattr(entry, "links", []):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
            break

    doi = getattr(entry, "arxiv_doi", None)

    published_date = _parse_date(entry.published)
    updated_date = _parse_date(entry.updated) if hasattr(entry, "updated") else None

    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(entry.title.split()),  # collapse whitespace/newlines
        "abstract": " ".join(entry.summary.split()),
        "authors": authors,
        "categories": categories,
        "primary_category": primary_category,
        "published_date": published_date,
        "updated_date": updated_date,
        "pdf_url": pdf_url,
        "doi": doi,
    }


def _parse_date(date_str: str) -> datetime:
    """arXiv dates look like '2024-01-15T18:30:00Z'."""
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")