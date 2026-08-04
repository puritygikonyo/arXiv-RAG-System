"""
Clear ALL papers from Postgres and OpenSearch — full wipe, not scoped to
one category. Reuses the same delete_papers() logic from
remove_category.py (chunks removed from OpenSearch first, then papers
index, then Postgres — same ordering, same safety guarantees).

SAFE BY DEFAULT: without --confirm, this only shows how many papers would
be deleted. Nothing is deleted until you pass --confirm explicitly.

Usage:
    # See what would be deleted (safe, no changes made)
    uv run python scripts/clear_all_papers.py

    # Actually delete everything
    uv run python scripts/clear_all_papers.py --confirm
"""

import argparse
import asyncio

from sqlalchemy import select

from src.database import AsyncSessionLocal
from src.logger import get_logger, setup_logging
from src.models import Paper

# Reuse the exact same deletion logic used for category-scoped removal —
# ordering (OpenSearch chunks -> OpenSearch papers -> Postgres cascade)
# and error handling stay identical.
from remove_category import delete_papers

setup_logging()
logger = get_logger(__name__)


async def find_all_papers() -> list[str]:
    async with AsyncSessionLocal() as session:
        result = await session.scalars(select(Paper.arxiv_id))
        return list(result.all())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Delete ALL papers from the system")
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually delete. Without this, only shows what would be deleted.",
    )
    args = parser.parse_args()

    arxiv_ids = await find_all_papers()

    if not arxiv_ids:
        print("No papers found. Nothing to delete.")
        return

    print(f"Found {len(arxiv_ids)} total papers in the system:")
    for aid in arxiv_ids[:10]:
        print(f"  {aid}")
    if len(arxiv_ids) > 10:
        print(f"  ... and {len(arxiv_ids) - 10} more")

    if not args.confirm:
        print(
            f"\nDRY RUN — nothing deleted. Re-run with --confirm to actually "
            f"delete all {len(arxiv_ids)} papers and their chunks from "
            f"both Postgres and OpenSearch."
        )
        return

    print(f"\nDeleting {len(arxiv_ids)} papers and their chunks...")
    await delete_papers(arxiv_ids)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())