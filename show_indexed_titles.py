"""
Show a sample of indexed paper titles — use one of these as a guaranteed
real search term when testing /api/v1/hybrid-search.

Usage:
    uv run python show_indexed_titles.py

The output will be used in guiding the golden test by guiding the scripts of a guaranteed output
"""

import asyncio

from sqlalchemy import select

from src.database import AsyncSessionLocal
from src.models import IngestionStatus, Paper


async def show() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.scalars(
            select(Paper)
            .where(Paper.ingestion_status == IngestionStatus.indexed)
            .limit(15)
        )
        papers = result.all()

        if not papers:
            print("No indexed papers found yet.")
            return

        print(f"Sample of {len(papers)} indexed papers:\n")
        for p in papers:
            print(f"  [{p.primary_category}] {p.title}")


if __name__ == "__main__":
    asyncio.run(show())


    