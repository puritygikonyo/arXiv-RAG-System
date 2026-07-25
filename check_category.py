"""
Check papers by actual category (primary or cross-listed), not keyword
guessing — more reliable than searching title/abstract text.

Usage:
    uv run python check_category.py eess.SY
"""

import argparse
import asyncio

from sqlalchemy import select

from src.database import AsyncSessionLocal
from src.models import Paper


async def check(category: str) -> None:
    async with AsyncSessionLocal() as session:
        primary_result = await session.scalars(
            select(Paper).where(Paper.primary_category == category)
        )
        primary_matches = primary_result.all()

        all_result = await session.scalars(select(Paper))
        all_papers = all_result.all()
        cross_listed = [p for p in all_papers if category in (p.categories or [])]

        print(f"Papers with '{category}' as PRIMARY category: {len(primary_matches)}")
        print(f"Papers with '{category}' anywhere (incl. cross-listed): {len(cross_listed)}")

        print("\nSample titles:")
        for p in cross_listed[:10]:
            print(f"  [{p.primary_category}] {p.title}  (status: {p.ingestion_status.value})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    args = parser.parse_args()
    asyncio.run(check(args.category))
    