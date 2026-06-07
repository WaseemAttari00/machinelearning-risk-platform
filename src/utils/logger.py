"""
Centralized logging setup using Loguru.

Why Loguru instead of Python's standard `logging` module?
  - Standard logging requires ~10 lines of boilerplate to configure correctly.
  - Loguru gives you colored, formatted output with one import.
  - It supports structured logging and file rotation out of the box.
  - The API is cleaner: logger.info("msg") vs logging.getLogger(__name__).info("msg")

Usage in any other module:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loading dataset from {path}", path=cfg["data"]["raw_path"])
"""

import sys
from loguru import logger


def get_logger(name: str):
    """
    Return a configured Loguru logger bound with the caller's module name.

    The module name appears in every log line so you can immediately tell
    which part of the pipeline produced a message.

    Example output:
        2024-01-15 10:23:45 | INFO | src.data.ingestion | Loaded 150000 rows
    """
    # Remove the default Loguru handler (it has no module name in the format)
    logger.remove()

    # Add a new handler with our preferred format:
    # - Time with millisecond precision
    # - Log level (INFO, WARNING, ERROR) — Loguru colors these automatically
    # - Module name bound via .bind()
    # - The actual log message
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{extra[module]}</cyan> | "
               "<level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # Also write to a log file for debugging purposes.
    # rotation="10 MB" means a new file is created once the current one hits 10 MB.
    # retention="7 days" means log files older than a week are deleted automatically.
    logger.add(
        "logs/pipeline.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[module]} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        colorize=False,
    )

    return logger.bind(module=name)
