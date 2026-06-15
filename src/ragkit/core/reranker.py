"""Rerank adapter for DashScope's gte-rerank model.

Uses the LlamaIndex DashScope postprocessor so the interface is standardized
(swap to BGE/Cohere/Jina by replacing the implementation only).

Model selection:
    The DashScope `gte-rerank` (v1) endpoint was restricted with 403 AccessDenied
    in mid-2025 — `gte-rerank-v2` is the current usable model. Override via
    the RAG_RERANK_MODEL environment variable if a different model is wanted.
"""

from __future__ import annotations

import os

import numpy as np
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

from ragkit.config import get_config
from ragkit.logger import logger


# DashScope deprecated/restricted gte-rerank v1; default to v2.
_DEFAULT_RERANK_MODEL = "gte-rerank-v2"


def rerank_scores(query: str, texts: list[str]) -> np.ndarray:
    """Score each text against the query. Returns scores in the SAME order as input."""
    if not texts:
        return np.array([])

    cfg = get_config()
    cfg.require_api_key()

    model = os.getenv("RAG_RERANK_MODEL", _DEFAULT_RERANK_MODEL)

    # Tag each input with its position so scores can be re-aligned by a stable
    # id, not by text. Aligning by text is wrong: duplicate texts collide (one
    # overwrites the other), and any non-identical returned string silently
    # scores 0.0.
    nodes = [
        NodeWithScore(node=Node(text=t, metadata={"rerank_id": i}), score=1.0)
        for i, t in enumerate(texts)
    ]
    reranker = DashScopeRerank(
        model=model,
        top_n=len(texts),
        api_key=cfg.dashscope_api_key,
    )
    reranked = reranker.postprocess_nodes(nodes, query_str=query)

    # Re-align by rerank_id (input position); the reranker may reorder nodes.
    scores = np.full(len(texts), np.nan, dtype=float)
    for nws in reranked:
        idx = nws.node.metadata.get("rerank_id")
        if isinstance(idx, int) and 0 <= idx < len(texts) and nws.score is not None:
            scores[idx] = nws.score

    missing = np.isnan(scores)
    if missing.any():
        # Never silently zero a missing score: warn and fall back to the lowest
        # observed score so an unscored chunk sinks instead of masquerading as a
        # real 0.0 relevance.
        observed = scores[~missing]
        fallback = float(observed.min()) if observed.size else 0.0
        logger.warning(
            "Reranker returned no score for %d/%d input(s); falling back to %.4f",
            int(missing.sum()),
            len(texts),
            fallback,
        )
        scores[missing] = fallback

    return scores
