"""Rerank adapter for DashScope's gte-rerank model.

Uses the LlamaIndex DashScope postprocessor so the interface is standardized
(swap to BGE/Cohere/Jina by replacing the implementation only).
"""

from __future__ import annotations

import numpy as np
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

from ragkit.config import get_config


def rerank_scores(query: str, texts: list[str]) -> np.ndarray:
    """Score each text against the query. Returns scores in the SAME order as input."""
    if not texts:
        return np.array([])

    cfg = get_config()
    cfg.require_api_key()

    nodes = [NodeWithScore(node=Node(text=t), score=1.0) for t in texts]
    reranker = DashScopeRerank(top_n=len(texts), api_key=cfg.dashscope_api_key)
    reranked = reranker.postprocess_nodes(nodes, query_str=query)

    # The reranker may return nodes in a different order — we re-align by text content.
    text_to_score = {n.node.text: n.score for n in reranked}
    return np.array([text_to_score.get(t, 0.0) for t in texts])
