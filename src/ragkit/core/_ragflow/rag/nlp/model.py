"""Compatibility shim used by search_v2.py.

The retrieval engine inside rag/nlp/search_v2.py imports two functions:
- generate_embedding(text)        — embed a single query string
- rerank_similarity(query, texts) — produce per-text scores

We delegate both to ragkit.core.embedder / ragkit.core.reranker so the
adapters live in one place.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from ragkit.core.embedder import embed_batch, embed_one
from ragkit.core.reranker import rerank_scores


def generate_embedding(
    text: Union[str, List[str]],
    api_key: str | None = None,  # kept for backward signature compatibility
    base_url: str | None = None,
    model_name: str | None = None,
    dimensions: int | None = None,
    encoding_format: str = "float",
    max_batch_size: int = 10,
) -> Union[List[float], List[List[float]]]:
    """Return the embedding for a single string or a list of strings."""
    if isinstance(text, str):
        return embed_one(text)
    return embed_batch(text)


def rerank_similarity(query: str, texts: List[str]):
    """Return (scores, None) — second element kept for legacy unpacking."""
    return rerank_scores(query, texts), None
