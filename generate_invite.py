"""
Generate a new invite token — run this once per person/school you want
to grant access to.

Usage:
    uv run python generate_invite.py "Lincoln High - Ms. Chen"
    uv run python generate_invite.py "Lincoln High - Ms. Chen" --limit 20

The token is what you share. The person enters it as the PASSWORD on the
Gradio login screen (username can be anything, e.g. their name).
"""

import argparse
import asyncio
import secrets

from src.database import AsyncSessionLocal
from src.models import Invite


async def create_invite(label: str, daily_limit: int | None) -> None:
    token = secrets.token_urlsafe(16)  # short, URL-safe, hard to guess

    async with AsyncSessionLocal() as session:
        invite = Invite(token=token, label=label, daily_question_limit=daily_limit)
        session.add(invite)
        await session.commit()

    print(f"\nInvite created for: {label}")
    print(f"Token (share this as their password): {token}")
    if daily_limit:
        print(f"Daily question limit: {daily_limit}")
    print("\nTell them: go to <your Gradio URL>, enter any username, "
          f"and use this token as the password: {token}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a new invite token")
    parser.add_argument("label", help="Who this invite is for, e.g. 'Lincoln High - Ms. Chen'")
    parser.add_argument("--limit", type=int, default=None, help="Optional daily question limit")
    args = parser.parse_args()

    asyncio.run(create_invite(args.label, args.limit))