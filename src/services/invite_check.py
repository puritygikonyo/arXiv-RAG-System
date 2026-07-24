"""
Invite enforcement — checks that an invite token is valid, not revoked,
and under its daily question limit, before letting a request run.

Used by /api/v1/ask to reject blocked/over-limit users early, before any
LLM/search work happens (so a revoked or exhausted invite costs nothing).
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from src.database import AsyncSessionLocal
from src.logger import get_logger
from src.models import Invite, QueryLog

logger = get_logger(__name__)


async def check_invite_allowed(invite_token: str | None) -> tuple[bool, str]:
    """
    Returns (allowed, reason). reason is only meaningful when not allowed —
    it's the message shown to the user, so keep it short and friendly.

    A missing/empty token is treated as not allowed — every request going
    through the invite-gated UI should be carrying one. (If you also want
    to support fully open, ungated access via a direct API call, handle
    that separately rather than loosening this check.)
    """
    if not invite_token:
        return False, "No access token provided. Please log in again."

    async with AsyncSessionLocal() as session:
        invite = await session.scalar(select(Invite).where(Invite.token == invite_token))

        if invite is None:
            return False, "Access token not recognized. Please check your invite."

        if invite.revoked:
            return False, "This access token has been revoked. Contact the admin for access."

        if invite.daily_question_limit is not None:
            since = datetime.now(UTC) - timedelta(hours=24)
            count = await session.scalar(
                select(func.count(QueryLog.id)).where(
                    QueryLog.invite_token == invite_token,
                    QueryLog.created_at >= since,
                )
            )
            if count is not None and count >= invite.daily_question_limit:
                return False, (
                    f"You've reached your daily limit of "
                    f"{invite.daily_question_limit} questions. Try again tomorrow."
                )

        # Stamp usage for visibility (first_used_at / last_used_at), same
        # bookkeeping the login check does — question-time is a more
        # accurate "last active" signal than login time alone.
        now = datetime.now(UTC)
        if invite.first_used_at is None:
            invite.first_used_at = now
        invite.last_used_at = now
        await session.commit()

    return True, ""