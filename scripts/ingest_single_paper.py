"""
ingest_single_paper.py — ingest one arXiv paper by ID, outside the normal
category-sweep flow in ingest_arxiv.py.

Reuses the real pipeline's functions (parsing, Postgres save, OpenSearch
indexing, embeddings) instead of reimplementing them, so a single-paper
test run stays consistent with normal ingestion.

USAGE:
    uv run python ingest_single_paper.py 1706.03762
    uv run python ingest_single_paper.py            # defaults to 1706.03762

Point your .env at the same Aiven (OpenSearch) + Neon (Postgres) you use
for normal ingestion before running this.
"""

import asyncio
import sys

import feedparser
import httpx

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

# Reuse the real parsing + save logic instead of duplicating it.
from ingest_arxiv import _parse_entry, save_paper_to_postgres, mark_indexed

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

DEFAULT_ARXIV_ID = "1706.03762"  # Attention Is All You Need


def fetch_paper_by_id(arxiv_id: str) -> dict:
    """Query arXiv's API for a single paper by ID (id_list, not cat:)."""
    params = {"id_list": arxiv_id}

    logger.info("fetching_arxiv_paper", arxiv_id=arxiv_id)

    response = httpx.get(
        settings.arxiv_api_base_url, params=params, timeout=30, follow_redirects=True
    )
    response.raise_for_status()

    feed = feedparser.parse(response.text)

    if not feed.entries:
        raise ValueError(f"No arXiv entry found for id {arxiv_id}")

    entry = feed.entries[0]
    paper = _parse_entry(entry)

    logger.info("fetched_arxiv_paper", arxiv_id=paper["arxiv_id"], title=paper["title"])
    return paper


async def run_single_ingestion(arxiv_id: str) -> None:
    logger.info("single_paper_ingestion_starting", arxiv_id=arxiv_id)

    await init_opensearch()
    ensure_chunks_index_exists()

    paper = fetch_paper_by_id(arxiv_id)

    is_new = await save_paper_to_postgres(paper)
    if not is_new:
        logger.info(
            "paper_already_exists",
            arxiv_id=paper["arxiv_id"],
            note="skipping re-fetch, but still checking index status below",
        )

    # Same "check what's actually unindexed" approach as the main script,
    # rather than assuming is_new tells the whole story.
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select

        db_paper = await session.scalar(
            select(Paper).where(Paper.arxiv_id == paper["arxiv_id"])
        )
        already_indexed = db_paper is not None and db_paper.ingestion_status == IngestionStatus.indexed

    if already_indexed:
        logger.info("paper_already_indexed", arxiv_id=paper["arxiv_id"])
        await close_opensearch()
        return

    paper_for_index = {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper["authors"],
        "categories": paper["categories"],
        "primary_category": paper["primary_category"],
        "published_at": paper["published_date"].isoformat(),
        "pdf_url": paper["pdf_url"],
    }

    result = bulk_index_papers([paper_for_index])
    logger.info("paper_metadata_index_complete", **result)

    try:
        chunk_result = index_paper_with_embeddings(paper_for_index)
        if chunk_result["errors"] == 0:
            await mark_indexed(paper["arxiv_id"], IngestionStatus.indexed)
            logger.info("paper_fully_indexed", arxiv_id=paper["arxiv_id"])
        else:
            await mark_indexed(paper["arxiv_id"], IngestionStatus.failed)
            logger.warning(
                "paper_indexed_with_errors",
                arxiv_id=paper["arxiv_id"],
                errors=chunk_result["errors"],
            )
    except Exception as e:
        await mark_indexed(paper["arxiv_id"], IngestionStatus.failed)
        logger.error(
            "embedding_pipeline_failed", arxiv_id=paper["arxiv_id"], error=str(e)
        )

    await close_opensearch()
    logger.info("single_paper_ingestion_complete", arxiv_id=paper["arxiv_id"])


if __name__ == "__main__":
    target_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARXIV_ID
    asyncio.run(run_single_ingestion(target_id))