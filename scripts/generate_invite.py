"""
Generate a new invite — run this once per person/school you want to
grant access to.

Usage:
    uv run python generate_invite.py mchen "Lincoln High - Ms. Chen"
    uv run python generate_invite.py mchen "Lincoln High - Ms. Chen" --limit 20

Login convention: username is what you assign here (share it as-is),
password is the generated token (the real secret — both are checked).
"""

import argparse
import asyncio
import secrets

from src.database import AsyncSessionLocal
from src.models import Invite


async def create_invite(username: str, label: str, daily_limit: int | None) -> None:
    token = secrets.token_urlsafe(16)

    async with AsyncSessionLocal() as session:
        invite = Invite(
            username=username, token=token, label=label, daily_question_limit=daily_limit
        )
        session.add(invite)
        await session.commit()

    print(f"\nInvite created for: {label}")
    print(f"Username: {username}")
    print(f"Password (token): {token}")
    if daily_limit:
        print(f"Daily question limit: {daily_limit}")
    print("\nShare both the username and password with them — see the email template.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a new invite")
    parser.add_argument("username", help="Login username to assign, e.g. 'mchen'")
    parser.add_argument("label", help="Who this invite is for, e.g. 'Lincoln High - Ms. Chen'")
    parser.add_argument("--limit", type=int, default=None, help="Optional daily question limit")
    args = parser.parse_args()

    asyncio.run(create_invite(args.username, args.label, args.limit))