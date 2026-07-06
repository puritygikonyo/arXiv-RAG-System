"""
Database seeding script — Phase 3 will populate this with real data.

Usage:
    make seed
    # or:
    uv run python scripts/seed_db.py
"""

import asyncio

from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


async def main() -> None:
    logger.info("seed_script_placeholder", message="Seed logic will be added in Phase 3")
    print("✓ Seed script placeholder — will be implemented in Phase 3 (PostgreSQL Models)")


if __name__ == "__main__":
    asyncio.run(main())
