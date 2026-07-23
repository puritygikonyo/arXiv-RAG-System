"""
arXiv ingestion script — the missing piece connecting arXiv's API to your
existing chunking/embedding/indexing pipeline.

PARSING LOGIC: ported from the Airflow project's include/arxiv_client.py
  (feedparser-based — more robust than manual XML parsing, and already
  proven against arXiv's feed quirks). Field names match the Paper model
  exactly: published_date, updated_date, doi.

STORAGE: uses this app's own async SQLAlchemy setup (AsyncSessionLocal),
  not the Airflow DAG's sync psycopg2 — avoids adding a new dependency and
  keeps this script self-contained within the main app's environment.

WHY A STANDALONE SCRIPT, NOT AIRFLOW:
  Running Airflow/Astronomer just to trigger an occasional ingestion is
  infrastructure you don't need yet at this scale. This script reuses the
  same tested fetch logic without requiring a scheduler to be hosted.
  The same fetch_papers_for_category() can be wired into the real DAG
  later if scheduled re-ingestion becomes worth it.

NEW DEPENDENCY: feedparser (not currently in pyproject.toml)
    uv add feedparser

USAGE:
    uv run python ingest_arxiv.py

  Point your .env at Aiven (OpenSearch) + Neon (Postgres) before running
  this against production data.

ARXIV RATE LIMIT: arXiv's docs ask for a delay between requests — we sleep
  between category fetches to respect this.
"""

import asyncio
import time
from datetime import datetime

import feedparser
import httpx
from sqlalchemy import select

from src.config import get_settings
from src.database import AsyncSessionLocal
from src.logger import get_logger, setup_logging
from src.models import IngestionStatus, Paper
from src.services.embeddings.vector_indexer import (
    ensure_chunks_index_exists,
    index_paper_with_embeddings,
)
from src.services.search.client import close_opensearch, init_opensearch
from src.services.search.indexer import bulk_index_papers

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

ARXIV_REQUEST_DELAY_SECONDS = 3


def _parse_date(date_str: str) -> datetime:
    """arXiv dates look like '2024-01-15T18:30:00Z'."""
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")


def _parse_entry(entry) -> dict:
    """
    Convert a single feedparser entry into a dict matching the Paper model.
    Ported from the Airflow project's include/arxiv_client.py — field names
    (published_date, updated_date, doi) match Paper's columns exactly.
    """
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
        "title": " ".join(entry.title.split()),
        "abstract": " ".join(entry.summary.split()),
        "authors": authors,
        "categories": categories,
        "primary_category": primary_category,
        "published_date": published_date,
        "updated_date": updated_date,
        "pdf_url": pdf_url,
        "doi": doi,
    }


def fetch_papers_for_category(category: str, max_results: int) -> list[dict]:
    """Query arXiv's API for one category. Same logic as the Airflow DAG's client."""
    params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }

    logger.info("fetching_arxiv_category", category=category, max_results=max_results)

    response = httpx.get(
        settings.arxiv_api_base_url, params=params, timeout=30, follow_redirects=True
    )
    response.raise_for_status()

    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        try:
            papers.append(_parse_entry(entry))
        except Exception as e:
            logger.warning(
                "skipping_malformed_entry",
                entry_id=getattr(entry, "id", "unknown"),
                error=str(e),
            )
            continue

    logger.info("fetched_arxiv_category", category=category, papers_found=len(papers))
    return papers


async def save_paper_to_postgres(paper: dict) -> bool:
    """
    Save a paper to Postgres if it doesn't already exist.
    Returns True if newly inserted (so we know to index it), False if it
    already existed (skip re-indexing — same de-dup behaviour as the DAG's
    ON CONFLICT DO NOTHING).
    """
    async with AsyncSessionLocal() as session:
        existing = await session.scalar(
            select(Paper).where(Paper.arxiv_id == paper["arxiv_id"])
        )
        if existing:
            return False

        db_paper = Paper(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            abstract=paper["abstract"],
            authors=paper["authors"],
            categories=paper["categories"],
            primary_category=paper["primary_category"],
            published_date=paper["published_date"],
            updated_date=paper["updated_date"],
            pdf_url=paper["pdf_url"],
            doi=paper["doi"],
            ingestion_status=IngestionStatus.fetched,
        )
        session.add(db_paper)
        await session.commit()
        return True


async def mark_indexed(arxiv_id: str, status: IngestionStatus) -> None:
    async with AsyncSessionLocal() as session:
        paper = await session.scalar(select(Paper).where(Paper.arxiv_id == arxiv_id))
        if paper:
            paper.ingestion_status = status
            await session.commit()


async def run_ingestion() -> None:
    logger.info("ingestion_starting", categories=settings.arxiv_categories)

    await init_opensearch()
    ensure_chunks_index_exists()

    # De-duplicate across categories, same as the DAG (a paper can be
    # cross-listed in multiple categories).
    fetched_by_id: dict[str, dict] = {}

    for i, category in enumerate(settings.arxiv_categories):
        papers = fetch_papers_for_category(
            category, settings.arxiv_max_results_per_run
        )
        for paper in papers:
            fetched_by_id[paper["arxiv_id"]] = paper

        if i < len(settings.arxiv_categories) - 1:
            time.sleep(ARXIV_REQUEST_DELAY_SECONDS)

    logger.info("fetch_complete", unique_papers=len(fetched_by_id))

    # Save to Postgres, tracking which ones are genuinely new
    new_papers: list[dict] = []
    for paper in fetched_by_id.values():
        is_new = await save_paper_to_postgres(paper)
        if is_new:
            new_papers.append(paper)

    logger.info("postgres_save_complete", new_papers=len(new_papers))

    # Don't rely solely on "newly inserted this run" — a paper can exist in
    # Postgres but never have been indexed (e.g. an earlier run saved it
    # then failed before reaching OpenSearch, as happened during setup
    # today). Query for ANY paper not yet marked indexed instead.
    #
    # INDEX_BATCH_LIMIT caps how many pending papers this run processes —
    # useful for a fast test run against a large backlog. Set to None to
    # process everything pending in one run.
    INDEX_BATCH_LIMIT = 250

    async with AsyncSessionLocal() as session:
        query = select(Paper).where(Paper.ingestion_status != IngestionStatus.indexed)
        if INDEX_BATCH_LIMIT:
            query = query.limit(INDEX_BATCH_LIMIT)
        result_rows = await session.scalars(query)
        unindexed = result_rows.all()

    papers_to_index = [
        {
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "authors": p.authors,
            "categories": p.categories,
            "primary_category": p.primary_category,
            "published_at": p.published_date.isoformat(),
            "pdf_url": p.pdf_url,
        }
        for p in unindexed
    ]

    logger.info("papers_pending_index", count=len(papers_to_index))

    if not papers_to_index:
        logger.info("no_papers_to_index")
        await close_opensearch()
        return

    result = bulk_index_papers(papers_to_index)
    logger.info("papers_index_complete", **result)

    chunk_errors = 0
    for paper in papers_to_index:
        try:
            chunk_result = index_paper_with_embeddings(paper)
            if chunk_result["errors"] == 0:
                await mark_indexed(paper["arxiv_id"], IngestionStatus.indexed)
            else:
                chunk_errors += chunk_result["errors"]
        except Exception as e:
            logger.error(
                "embedding_pipeline_failed", arxiv_id=paper["arxiv_id"], error=str(e)
            )
            await mark_indexed(paper["arxiv_id"], IngestionStatus.failed)

    logger.info(
        "ingestion_complete",
        total_new_papers=len(new_papers),
        chunk_errors=chunk_errors,
    )

    await close_opensearch()


if __name__ == "__main__":
    asyncio.run(run_ingestion())