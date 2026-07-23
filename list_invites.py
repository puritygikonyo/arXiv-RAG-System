"""
List all invites currently in Postgres — confirms whether
generate_invite.py actually wrote a row, and shows the exact token stored.

Usage:
    uv run python list_invites.py
"""

import asyncio

from sqlalchemy import select

from src.database import AsyncSessionLocal
from src.models import Invite


async def list_invites() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.scalars(select(Invite))
        invites = result.all()

        if not invites:
            print("No invites found in the database.")
            return

        for inv in invites:
            print(
                f"label={inv.label!r} token={inv.token!r} "
                f"revoked={inv.revoked} limit={inv.daily_question_limit}"
            )


if __name__ == "__main__":
    asyncio.run(list_invites())