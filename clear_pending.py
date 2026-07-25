"""
Remove all papers that were fetched but never indexed (status != indexed).

Since these were never written to OpenSearch, this is Postgres-only —
nothing to clean up on the search side.

SAFE BY DEFAULT: without --confirm, only shows what would be deleted.

Usage:
    uv run python clear_pending.py            # dry run
    uv run python clear_pending.py --confirm  # actually delete
"""

import argparse
import asyncio

from sqlalchemy import select

from src.database import AsyncSessionLocal
from src.models import IngestionStatus, Paper


async def main() -> None:
    parser = argparse.ArgumentParser(description="Clear pending (never-indexed) papers")
    parser.add_argument("--confirm", action="store_true", help="Actually delete")
    args = parser.parse_args()

    async with AsyncSessionLocal() as session:
        result = await session.scalars(
            select(Paper).where(Paper.ingestion_status != IngestionStatus.indexed)
        )
        pending = result.all()

        if not pending:
            print("No pending papers found.")
            return

        print(f"Found {len(pending)} pending (never-indexed) papers.")

        if not args.confirm:
            print("\nDRY RUN — nothing deleted. Re-run with --confirm to delete.")
            return

        for paper in pending:
            await session.delete(paper)
        await session.commit()

    print(f"Deleted {len(pending)} pending papers from Postgres.")


if __name__ == "__main__":
    asyncio.run(main())   