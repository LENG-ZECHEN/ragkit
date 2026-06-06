"""Retrieval metrics: Recall@K, MRR, nDCG@10 from one ``EvalTrace`` + ``QAItem``.

Refusal-item handling (gt is the empty list)
--------------------------------------------
Refusal questions intentionally have NO relevant chunks; recall/MRR/nDCG are
mathematically undefined (0/0). After considering three options:

  (A) all metrics = 1.0 iff retrieved is also empty, else 0.0
  (B) all metrics = NaN with a separate ``refusal_correct: bool`` flag
  (C) silently return 0.0 everywhere

we chose **(A) with a refusal_correct flag** — a hybrid:
  * ``refusal_correct = True``  ⇔  ``retrieved`` (after chunk-filter) is empty.
  * On a correct refusal we report metrics = 1.0 across the board (so a sweep
    average isn't dragged down by a perfectly-handled refusal).
  * On a failed refusal we report metrics = 0.0 (retrieval surfaced something
    when nothing was relevant — a precision failure).

Rationale: aggregation is easier with concrete floats than with NaN, and the
explicit ``refusal_correct`` flag lets the sweep report split refusal cases
out when desired.

Only items where ``kind == "chunk"`` are considered relevant. Entities,
communities, relations, and points are filtered out before computing metrics,
because the ground-truth annotations in the dataset are chunk-ids only.
"""

from __future__ import annotations

import math
from typing import TypedDict

from .schema import QAItem


# --------------------------------------------------------------------------
# Public TypedDict
# --------------------------------------------------------------------------


class RetrievalMetrics(TypedDict):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    refusal_correct: bool


# Ks for Recall@K — keep this in one place so a future grid-change is one edit.
_RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _chunk_ids_in_order(trace: dict) -> list[str]:
    """Return chunk_ids from ``trace["retrieved"]`` filtered to ``kind=="chunk"``.

    Order is preserved (it reflects the retriever's ranking). Items missing
    a ``kind`` key are treated as chunks (older traces, defensive default).
    """
    out: list[str] = []
    for item in trace.get("retrieved", []) or []:
        if item.get("kind", "chunk") == "chunk":
            out.append(item["chunk_id"])
    return out


def _recall_at_k(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:  # caller handles the refusal case separately.
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & gt) / len(gt)


def _mrr(retrieved: list[str], gt: set[str]) -> float:
    """1 / (rank of first relevant chunk). 0 if none of the gt appear."""
    if not gt:
        return 0.0
    for idx, cid in enumerate(retrieved, start=1):
        if cid in gt:
            return 1.0 / idx
    return 0.0


def _ndcg_at_10(retrieved: list[str], gt: set[str]) -> float:
    """Binary-relevance nDCG@10. Returns 0.0 if IDCG is 0."""
    if not gt:
        return 0.0
    # DCG over the top-10 retrieved items.
    dcg = 0.0
    for i, cid in enumerate(retrieved[:10], start=1):
        rel = 1.0 if cid in gt else 0.0
        # log2(i+1) — standard formula with rank i starting at 1.
        dcg += rel / math.log2(i + 1)
    # IDCG: best case is every relevant item ranked first, capped at 10.
    ideal_hits = min(len(gt), 10)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def compute_metrics(trace: dict, qa: QAItem) -> RetrievalMetrics:
    """Return Recall@{1,3,5,10}, MRR, nDCG@10 for one (trace, qa) pair.

    Handles refusal items (empty ``ground_truth_chunk_ids``) per the choice
    documented at the top of this module.
    """
    retrieved = _chunk_ids_in_order(trace)
    gt = set(qa.ground_truth_chunk_ids)

    # Refusal-case branch.
    if not gt:
        correct = len(retrieved) == 0
        score = 1.0 if correct else 0.0
        return RetrievalMetrics(
            recall_at_1=score,
            recall_at_3=score,
            recall_at_5=score,
            recall_at_10=score,
            mrr=score,
            ndcg_at_10=score,
            refusal_correct=correct,
        )

    # Normal case.
    return RetrievalMetrics(
        recall_at_1=_recall_at_k(retrieved, gt, 1),
        recall_at_3=_recall_at_k(retrieved, gt, 3),
        recall_at_5=_recall_at_k(retrieved, gt, 5),
        recall_at_10=_recall_at_k(retrieved, gt, 10),
        mrr=_mrr(retrieved, gt),
        ndcg_at_10=_ndcg_at_10(retrieved, gt),
        refusal_correct=False,  # N/A for non-refusal items.
    )
