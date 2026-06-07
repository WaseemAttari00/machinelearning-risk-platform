"""Structured logging via Loguru with colored console output and rotating file sink."""

import sys
from loguru import logger


def get_logger(name: str):
    """
    Return a Loguru logger bound with the caller's module name.

    Writes to stderr (colored) and logs/pipeline.log (rotating at 10 MB).
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{extra[module]}</cyan> | "
               "<level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    logger.add(
        "logs/pipeline.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[module]} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        colorize=False,
    )

    return logger.bind(module=name)
