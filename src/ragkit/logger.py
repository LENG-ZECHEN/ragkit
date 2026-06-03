"""Colored logger for ragkit."""

from __future__ import annotations

import logging
import os

import colorlog


def setup_logger(name: str = "ragkit", level: str | int | None = None) -> logging.Logger:
    """Build a colorized logger. Idempotent; safe to call multiple times."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = level or os.getenv("RAG_LOG_LEVEL", "INFO")
    logger.setLevel(level)

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()
