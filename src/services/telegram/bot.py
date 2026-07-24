"""
Telegram bot — forwards messages to the /api/v1/ask endpoint and relays
the answer back. Runs via long polling (no public URL required), as a
separate process from the FastAPI server and the Gradio UI.

Design: the bot is a thin HTTP client of your own API, not a second
implementation of the RAG pipeline. This means every message sent
through Telegram automatically gets the same semantic caching, Langfuse
tracing, and query logging as a normal /ask request — nothing to
duplicate or keep in sync.

ACCESS CONTROL: reuses the same `invites` table as the Gradio web UI,
instead of a static TELEGRAM_ALLOWED_CHAT_IDS list. A person links their
Telegram chat to an existing invite once via /register <token> (the same
token generate_invite.py prints). After that, every question is checked
against the same revocation/daily-limit rules as the web UI, via
check_invite_allowed() — one source of truth for both channels.
"""

import json
import os
import re

import httpx
from sqlalchemy import select
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.config import get_settings
from src.database import AsyncSessionLocal
from src.logger import get_logger
from src.models import Invite
from src.services.invite_check import check_invite_allowed

logger = get_logger(__name__)
settings = get_settings()

_default_ask_endpoint = f"http://{settings.api_host}:{settings.api_port}/api/v1/ask"
if settings.api_host == "0.0.0.0":
    _default_ask_endpoint = f"http://localhost:{settings.api_port}/api/v1/ask"

# Override with the deployed Render API URL when running as its own
# service, e.g. TELEGRAM_API_URL=https://arxiv-rag-system-xxxx.onrender.com/api/v1/ask
ASK_ENDPOINT = os.environ.get("TELEGRAM_API_URL", _default_ask_endpoint)

TELEGRAM_MAX_MESSAGE_LENGTH = 4000


async def _get_invite_by_chat_id(chat_id: int) -> Invite | None:
    async with AsyncSessionLocal() as session:
        return await session.scalar(
            select(Invite).where(Invite.telegram_chat_id == chat_id)
        )


async def _register_chat(token: str, chat_id: int) -> tuple[bool, str]:
    """
    Link a Telegram chat to an existing invite, identified by its token.
    Returns (success, message).
    """
    async with AsyncSessionLocal() as session:
        invite = await session.scalar(select(Invite).where(Invite.token == token))

        if invite is None:
            return False, "That token wasn't recognized. Double-check it and try again."

        if invite.revoked:
            return False, "That token has been revoked. Contact the admin for access."

        if invite.telegram_chat_id is not None and invite.telegram_chat_id != chat_id:
            return False, "That token is already linked to a different Telegram chat."

        invite.telegram_chat_id = chat_id
        await session.commit()

    return True, f"You're registered as {invite.label}. Send any research question to get started."


def _strip_markdown(text: str) -> str:
    """
    Convert common LLM markdown formatting (from the generator node's
    Groq output) into clean plain text for Telegram.

    We deliberately DON'T use parse_mode="MarkdownV2" here. Telegram's
    MarkdownV2 requires escaping a long list of special characters
    (. - ! ( ) etc.) in every piece of non-formatted text, and Groq's
    raw markdown output isn't guaranteed to already be valid MarkdownV2.
    A single unescaped character anywhere in the answer causes Telegram
    to reject the WHOLE message with a 400 error. Stripping formatting
    down to plain text instead is less pretty but can never fail to send.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)

    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`", r"\1", text)

    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    return text.strip()


def _split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """
    Split text into chunks that fit under Telegram's message length limit,
    breaking on paragraph boundaries where possible so a chunk never cuts
    a sentence in half if it can be avoided.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= max_length:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_length:
            current = paragraph
            continue

        words = paragraph.split(" ")
        piece = ""
        for word in words:
            if len(word) > max_length:
                if piece:
                    chunks.append(piece)
                    piece = ""
                for i in range(0, len(word), max_length):
                    chunks.append(word[i:i + max_length])
                continue

            candidate_piece = f"{piece} {word}" if piece else word
            if len(candidate_piece) <= max_length:
                piece = candidate_piece
            else:
                chunks.append(piece)
                piece = word
        current = piece

    if current:
        chunks.append(current)

    return chunks


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        "arXiv Research Assistant is online.\n\n"
        "To get access, register with the token you were given:\n"
        "/register YOUR_TOKEN\n\n"
        "Once registered, just send any research question."
    )
    logger.info("telegram_start_command", chat_id=chat_id)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        "arXiv Research Assistant — how to use this bot:\n\n"
        "If you haven't already, register with your invite token:\n"
        "/register YOUR_TOKEN\n\n"
        "Then just send any question about topics covered in the paper "
        "database (computer science, ML, physics, math, and related "
        "research fields). For example:\n"
        '"What is the attention mechanism?"\n\n'
        "The bot will:\n"
        "1. Check the question is answerable from the paper database\n"
        "2. Search for relevant paper excerpts\n"
        "3. Grade how relevant they are, retrying the search if needed\n"
        "4. Write an answer with source links\n\n"
        "This can take 15-40 seconds while it searches and reasons "
        "through the answer.\n\n"
        "Commands:\n"
        "/register YOUR_TOKEN — link this chat to your invite\n"
        "/start — welcome message\n"
        "/help — show this message"
    )
    logger.info("telegram_help_command", chat_id=chat_id)


async def register_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text("Usage: /register YOUR_TOKEN")
        return

    token = context.args[0].strip()
    success, message = await _register_chat(token, chat_id)
    await update.message.reply_text(message)
    logger.info("telegram_register_attempt", chat_id=chat_id, success=success)


async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forward the message text to /api/v1/ask and relay the answer back."""
    chat_id = update.effective_chat.id
    question = update.message.text

    invite = await _get_invite_by_chat_id(chat_id)
    if invite is None:
        await update.message.reply_text(
            "This chat isn't registered yet. Use /register YOUR_TOKEN first."
        )
        logger.warning("telegram_unregistered_attempt", chat_id=chat_id, question=question[:100])
        return

    allowed, reason = await check_invite_allowed(invite.token)
    if not allowed:
        await update.message.reply_text(reason)
        logger.warning("telegram_blocked", chat_id=chat_id, reason=reason)
        return

    logger.info("telegram_question_received", chat_id=chat_id, question=question[:100])

    await update.message.reply_text("Searching the paper database, one moment...")

    try:
        answer, citations = await _call_ask_endpoint(question, invite.token)
    except Exception as exc:
        logger.error("telegram_ask_failed", error=str(exc), chat_id=chat_id)
        await update.message.reply_text(
            "Something went wrong reaching the research system. Please try again."
        )
        return

    reply = _strip_markdown(answer)
    if citations:
        source_lines = "\n".join(f"https://arxiv.org/abs/{c}" for c in citations)
        reply += f"\n\nSources:\n{source_lines}"

    chunks = _split_message(reply)
    if len(chunks) > 1:
        logger.info(
            "telegram_reply_chunked",
            chat_id=chat_id,
            chunk_count=len(chunks),
            total_length=len(reply),
        )

    for chunk in chunks:
        await update.message.reply_text(chunk)


async def _call_ask_endpoint(question: str, invite_token: str) -> tuple[str, list[str]]:
    """
    Call /api/v1/ask and parse the SSE stream for the final answer.
    Passes invite_token so the API enforces the same revocation/limit
    rules — belt-and-suspenders alongside the check in ask_handler,
    since the API is the actual source of truth.
    """
    answer = "No answer was generated."
    citations: list[str] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        async with client.stream(
            "POST", ASK_ENDPOINT, json={"question": question, "invite_token": invite_token}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: "):])
                if event.get("node") == "blocked":
                    answer = event.get("reason", "Access denied.")
                    citations = []
                elif event.get("node") in ("generator", "reject") and "answer" in event:
                    answer = event["answer"]
                    citations = event.get("citations", [])

    return answer, citations


def build_application() -> Application:
    """Construct the Telegram Application with handlers registered."""
    if not settings.telegram_bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set in .env — "
            "create a bot via @BotFather and add the token first."
        )

    application = Application.builder().token(settings.telegram_bot_token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("register", register_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask_handler))

    return application