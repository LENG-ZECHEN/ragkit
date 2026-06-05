"""Embedding adapter.

Wraps DashScope's OpenAI-compatible embedding API. The DashScope free tier
caps batch size at 10, so we chunk requests transparently.
"""

from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from ragkit.config import get_config
from ragkit.logger import logger

# DashScope hard limit on inputs per embedding call.
_MAX_BATCH = 10


def _client() -> OpenAI:
    cfg = get_config()
    cfg.require_api_key()
    return OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url, timeout=cfg.llm_timeout)


def embed_one(text: str) -> list[float]:
    """Embed a single piece of text. Returns the vector directly."""
    cfg = get_config()
    resp = _client().embeddings.create(
        model=cfg.embedding_model,
        input=text,
        dimensions=cfg.embedding_dim,
        encoding_format="float",
    )
    return resp.data[0].embedding


def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    """Embed many texts. Automatically batches into chunks of _MAX_BATCH."""
    cfg = get_config()
    client = _client()
    texts = list(texts)

    out: list[list[float]] = []
    for i in range(0, len(texts), _MAX_BATCH):
        batch = texts[i : i + _MAX_BATCH]
        resp = client.embeddings.create(
            model=cfg.embedding_model,
            input=batch,
            dimensions=cfg.embedding_dim,
            encoding_format="float",
        )
        out.extend(item.embedding for item in resp.data)
        logger.debug(f"Embedded batch {i // _MAX_BATCH + 1} ({len(batch)} items)")
    return out
