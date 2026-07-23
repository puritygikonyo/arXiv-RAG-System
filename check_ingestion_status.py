"""
Quick check: how many papers are in Postgres, and how many are actually
indexed into OpenSearch (vs stuck in an earlier ingestion status).

Usage:
    uv run python check_ingestion_status.py
"""

import asyncio

from sqlalchemy import func, select

from src.database import AsyncSessionLocal
from src.models import IngestionStatus, Paper


async def check() -> None:
    async with AsyncSessionLocal() as session:
        total = await session.scalar(select(func.count(Paper.arxiv_id)))
        indexed = await session.scalar(
            select(func.count(Paper.arxiv_id)).where(
                Paper.ingestion_status == IngestionStatus.indexed
            )
        )
        failed = await session.scalar(
            select(func.count(Paper.arxiv_id)).where(
                Paper.ingestion_status == IngestionStatus.failed
            )
        )

        print(f"Total papers in Postgres: {total}")
        print(f"Indexed: {indexed}")
        print(f"Failed: {failed}")
        print(f"Not yet indexed (pending): {total - indexed - failed}")


if __name__ == "__main__":
    asyncio.run(check())
    