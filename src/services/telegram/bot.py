"""
Telegram bot — forwards messages to the /api/v1/ask endpoint and relays
the answer back. Runs via long polling (no public URL required), as a
separate process alongside the FastAPI server.

Design: the bot is a thin HTTP client of your own API, not a second
implementation of the RAG pipeline. This means every message sent
through Telegram automatically gets the same semantic caching, Langfuse
tracing, and query logging as a normal /ask request — nothing to
duplicate or keep in sync.
"""

import json
import re

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

ASK_ENDPOINT = f"http://{settings.api_host}:{settings.api_port}/api/v1/ask"
if settings.api_host == "0.0.0.0":
    # 0.0.0.0 means "listen on all interfaces" for the SERVER, but you
    # can't connect TO 0.0.0.0 as a client — use localhost instead.
    ASK_ENDPOINT = f"http://localhost:{settings.api_port}/api/v1/ask"

# Telegram hard-rejects any message body over 4096 characters. Leave a
# little headroom below the true limit for safety margin.
TELEGRAM_MAX_MESSAGE_LENGTH = 4000


def _is_allowed(chat_id: int) -> bool:
    """
    Check the chat against the allowlist.

    Empty allowlist = dev mode, anyone can use the bot. This is
    deliberately loud (logs a warning on every message) rather than
    silent, so an empty allowlist in production doesn't go unnoticed —
    same philosophy as the OpenSearch pool bug: a misconfiguration
    should be visible, not quietly wrong.
    """
    if not settings.telegram_allowed_chat_ids:
        logger.warning(
            "telegram_allowlist_empty",
            chat_id=chat_id,
            note="TELEGRAM_ALLOWED_CHAT_IDS is empty — allowing all chats. "
                 "Set it in .env to restrict access.",
        )
        return True
    return chat_id in settings.telegram_allowed_chat_ids


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
    # bold/italic markers: **text**, __text__, *text*, _text_
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)

    # inline code and fenced code blocks
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`", r"\1", text)

    # headers: "## Something" -> "Something"
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # markdown links: [text](url) -> "text (url)" -- Telegram auto-links
    # any bare URL in plain text, so the URL stays clickable
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    return text.strip()


def _split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """
    Split text into chunks that fit under Telegram's message length limit,
    breaking on paragraph boundaries where possible so a chunk never cuts
    a sentence in half if it can be avoided.

    Strategy:
      1. If the whole text already fits, return it as a single chunk.
      2. Otherwise split on blank lines (paragraphs) and pack them into
         chunks greedily, starting a new chunk whenever adding the next
         paragraph would exceed max_length.
      3. If a single paragraph is itself longer than max_length (rare,
         but possible with a dense citation list or no paragraph breaks
         at all), hard-split that paragraph on whitespace as a fallback
         so we never produce a chunk Telegram would reject outright.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        # +2 accounts for the "\n\n" that will rejoin this paragraph
        # onto `current` if it fits
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

        # single paragraph itself exceeds the limit — hard-split on words.
        # If a single "word" itself exceeds max_length (e.g. one huge
        # unbroken token with no spaces at all), fall back further to a
        # character-level slice so we never emit a chunk over the limit.
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
    """
    /start handler. Also surfaces the chat ID so the user can copy it
    into TELEGRAM_ALLOWED_CHAT_IDS in .env — there's no other easy way
    to discover your own chat ID before the allowlist exists.
    """
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        "arXiv Research Assistant is online.\n\n"
        f"Your chat ID is: {chat_id}\n"
        "Add this to TELEGRAM_ALLOWED_CHAT_IDS in .env to authorize this chat.\n\n"
        "Once authorized, just send any research question."
    )
    logger.info("telegram_start_command", chat_id=chat_id)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /help handler. Unlike /start (which is mainly for first-time setup and
    surfacing the chat ID), this is a quick reference for returning users
    who are already authorized and just want a reminder of what the bot does.
    """
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        "arXiv Research Assistant — how to use this bot:\n\n"
        "Just send any question about topics covered in the paper database "
        "(computer science, ML, physics, math, and related research fields). "
        "For example:\n"
        '"What is the attention mechanism?"\n\n'
        "The bot will:\n"
        "1. Check the question is answerable from the paper database\n"
        "2. Search for relevant paper excerpts\n"
        "3. Grade how relevant they are, retrying the search if needed\n"
        "4. Write an answer with source links\n\n"
        "This can take 15-40 seconds while it searches and reasons "
        "through the answer.\n\n"
        "Commands:\n"
        "/start — show your chat ID (needed for access authorization)\n"
        "/help — show this message"
    )
    logger.info("telegram_help_command", chat_id=chat_id)


async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forward the message text to /api/v1/ask and relay the answer back."""
    chat_id = update.effective_chat.id
    question = update.message.text

    if not _is_allowed(chat_id):
        await update.message.reply_text("This chat is not authorized to use this bot.")
        logger.warning("telegram_unauthorized_attempt", chat_id=chat_id, question=question[:100])
        return

    logger.info("telegram_question_received", chat_id=chat_id, question=question[:100])

    # Let the user know something's happening — full pipeline runs can
    # take 15-40s on a cache miss, and Telegram has no built-in
    # "typing..." indicator for long waits beyond a few seconds.
    await update.message.reply_text("Searching the paper database, one moment...")

    try:
        answer, citations = await _call_ask_endpoint(question)
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


async def _call_ask_endpoint(question: str) -> tuple[str, list[str]]:
    """
    Call /api/v1/ask and parse the SSE stream for the final answer.

    /ask streams progress events; the bot only cares about the final
    result, so this reads the whole stream and extracts the last
    "generator" (or "reject") event's answer + citations rather than
    relaying intermediate progress to Telegram.
    """
    answer = "No answer was generated."
    citations: list[str] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        async with client.stream(
            "POST", ASK_ENDPOINT, json={"question": question}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: "):])
                if event.get("node") in ("generator", "reject") and "answer" in event:
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
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask_handler))

    return application
