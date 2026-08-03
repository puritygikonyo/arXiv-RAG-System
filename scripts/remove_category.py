"""
Remove all papers belonging to a specific category — cleans up Postgres
AND both OpenSearch indices (papers + chunks) together, so nothing is
left orphaned or still searchable after "removal."

SAFE BY DEFAULT: without --confirm, this only shows you what WOULD be
deleted and how many papers/chunks are affected. Nothing is deleted
until you pass --confirm explicitly.

Usage:
    # See what would be deleted (safe, no changes made)
    uv run python remove_category.py eess.SY

    # Actually delete
    uv run python remove_category.py eess.SY --confirm

Matches on primary_category by default. Use --match-any-category to
also remove papers where the category appears anywhere in their
cross-listed categories list, not just as primary.
"""

import argparse
import asyncio

from sqlalchemy import select

from src.config import get_settings
from src.database import AsyncSessionLocal
from src.logger import get_logger, setup_logging
from src.models import Paper
from src.services.search.client import close_opensearch, get_opensearch_client, init_opensearch

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


async def find_matching_papers(category: str, match_any: bool) -> list[str]:
    async with AsyncSessionLocal() as session:
        if match_any:
            # categories is a JSON list column — filter in Python since
            # a JSON "contains" query varies by backend
            result = await session.scalars(select(Paper))
            all_papers = result.all()
            return [p.arxiv_id for p in all_papers if category in (p.categories or [])]
        else:
            result = await session.scalars(
                select(Paper.arxiv_id).where(Paper.primary_category == category)
            )
            return list(result.all())


async def delete_papers(arxiv_ids: list[str]) -> None:
    await init_opensearch()
    client = get_opensearch_client()

    # 1. Delete matching chunks from the chunks index (one paper -> many
    #    chunks, matched by arxiv_id field, not by chunk_id)
    client.delete_by_query(
        index=settings.opensearch_chunks_index,
        body={"query": {"terms": {"arxiv_id": arxiv_ids}}},
        refresh=True,
    )

    # 2. Delete the papers themselves from the papers index
    for arxiv_id in arxiv_ids:
        try:
            client.delete(index=settings.opensearch_papers_index, id=arxiv_id)
        except Exception as e:
            logger.warning("opensearch_paper_delete_failed", arxiv_id=arxiv_id, error=str(e))

    await close_opensearch()

    # 3. Delete from Postgres — Chunk rows cascade automatically via the
    #    ondelete="CASCADE" foreign key on Chunk.paper_id
    async with AsyncSessionLocal() as session:
        result = await session.scalars(select(Paper).where(Paper.arxiv_id.in_(arxiv_ids)))
        for paper in result.all():
            await session.delete(paper)
        await session.commit()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Remove papers by category")
    parser.add_argument("category", help="e.g. eess.SY")
    parser.add_argument(
        "--match-any-category", action="store_true",
        help="Also match if category is cross-listed, not just primary",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually delete. Without this, only shows what would be deleted.",
    )
    args = parser.parse_args()

    arxiv_ids = await find_matching_papers(args.category, args.match_any_category)

    if not arxiv_ids:
        print(f"No papers found matching category '{args.category}'.")
        return

    print(f"Found {len(arxiv_ids)} papers matching category '{args.category}':")
    for aid in arxiv_ids[:10]:
        print(f"  {aid}")
    if len(arxiv_ids) > 10:
        print(f"  ... and {len(arxiv_ids) - 10} more")

    if not args.confirm:
        print(
            f"\nDRY RUN — nothing deleted. Re-run with --confirm to actually "
            f"delete these {len(arxiv_ids)} papers and their chunks from "
            f"both Postgres and OpenSearch."
        )
        return

    print(f"\nDeleting {len(arxiv_ids)} papers and their chunks...")
    await delete_papers(arxiv_ids)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())