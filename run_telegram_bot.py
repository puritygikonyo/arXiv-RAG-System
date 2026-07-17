"""
Entry point for the Telegram bot. Runs as a separate long-lived process
from the FastAPI server — start uvicorn in one terminal, this in another.

Run:
    uv run python run_telegram_bot.py
"""

from src.logger import get_logger, setup_logging
from src.services.telegram.bot import build_application

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    application = build_application()
    logger.info("telegram_bot_starting")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()