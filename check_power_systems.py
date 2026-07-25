"""
Check whether any currently-indexed papers touch on Power Systems.

Usage:
    uv run python check_power_systems.py
"""

import asyncio

from sqlalchemy import or_, select

from src.database import AsyncSessionLocal
from src.models import Paper

KEYWORDS = ["power system", "power grid", "smart grid", "electrical grid", "power flow"]


async def check() -> None:
    async with AsyncSessionLocal() as session:
        conditions = [Paper.title.ilike(f"%{kw}%") for kw in KEYWORDS] + [
            Paper.abstract.ilike(f"%{kw}%") for kw in KEYWORDS
        ]
        result = await session.scalars(select(Paper).where(or_(*conditions)))
        papers = result.all()

        print(f"Found {len(papers)} papers mentioning power-systems terms:\n")
        for p in papers:
            print(f"  [{p.primary_category}] {p.title}")

        print("\nCurrently configured categories (from .env / config.py):")
        print("  cs.AI, cs.LG, cs.CL, cs.IR  (or whatever ARXIV_CATEGORIES is set to)")
        print("\nFor Power Systems specifically, add: eess.SY")


if __name__ == "__main__":
    asyncio.run(check())