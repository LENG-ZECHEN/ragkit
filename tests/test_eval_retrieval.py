"""Tests for ``evals.eval_retrieval.compute_metrics``.

Covers all three metrics × {refusal, no-hit, perfect-hit, partial-hit,
multi-gt} plus the ``kind != "chunk"`` filtering.
"""

from __future__ import annotations

import math

import pytest

from evals.eval_retrieval import compute_metrics
from evals.schema import QAItem


def _trace(chunk_ids: list[str], *, extras: list[dict] | None = None) -> dict:
    items: list[dict] = [
        {"chunk_id": cid, "rank": i + 1, "score": 1.0 - 0.01 * i, "kind": "chunk"}
        for i, cid in enumerate(chunk_ids)
    ]
    if extras:
        items.extend(extras)
    return {"retrieved": items}


def _qa(qa_id: str, gt: list[str], *, category: str = "factual") -> QAItem:
    return QAItem(
        id=qa_id, question="q", category=category,  # type: ignore[arg-type]
        ground_truth_chunk_ids=gt,
        gold_answer=None if category == "refusal" else "a",
    )


# -------- refusal branch --------


@pytest.mark.unit
def test_refusal_perfect_when_nothing_retrieved():
    m = compute_metrics(_trace([]), _qa("r-1", [], category="refusal"))
    assert m["refusal_correct"] is True
    for k in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
              "mrr", "ndcg_at_10"):
        assert m[k] == 1.0


@pytest.mark.unit
def test_refusal_failed_when_something_retrieved():
    m = compute_metrics(_trace(["aaa", "bbb"]), _qa("r-1", [], category="refusal"))
    assert m["refusal_correct"] is False
    for k in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
              "mrr", "ndcg_at_10"):
        assert m[k] == 0.0


@pytest.mark.unit
def test_refusal_only_non_chunk_items_counts_as_correct_refusal():
    """Entities/communities don't surface chunk content → still a refusal."""
    qa = _qa("r-1", [], category="refusal")
    trace = _trace([], extras=[
        {"chunk_id": "ent-1", "rank": 1, "score": 0.9, "kind": "entity"},
        {"chunk_id": "com-1", "rank": 2, "score": 0.8, "kind": "community"},
    ])
    m = compute_metrics(trace, qa)
    assert m["refusal_correct"] is True
    assert m["recall_at_1"] == 1.0


# -------- no-hit --------


@pytest.mark.unit
def test_no_hit_all_metrics_zero():
    m = compute_metrics(_trace(["miss-1", "miss-2"]), _qa("f-1", ["target"]))
    assert m["recall_at_1"] == 0.0
    assert m["recall_at_3"] == 0.0
    assert m["recall_at_5"] == 0.0
    assert m["recall_at_10"] == 0.0
    assert m["mrr"] == 0.0
    assert m["ndcg_at_10"] == 0.0
    assert m["refusal_correct"] is False


# -------- perfect / partial hit --------


@pytest.mark.unit
def test_perfect_hit_single_gt_at_rank_1():
    m = compute_metrics(_trace(["target", "x", "y"]), _qa("f-1", ["target"]))
    assert m["recall_at_1"] == 1.0
    assert m["recall_at_3"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ndcg_at_10"] == pytest.approx(1.0)


@pytest.mark.unit
def test_partial_hit_single_gt_at_rank_3():
    m = compute_metrics(_trace(["a", "b", "target", "c"]), _qa("f-1", ["target"]))
    assert m["recall_at_1"] == 0.0
    assert m["recall_at_3"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["mrr"] == pytest.approx(1.0 / 3.0)
    # DCG = 1/log2(4); IDCG = 1/log2(2) = 1.
    assert m["ndcg_at_10"] == pytest.approx(1.0 / math.log2(4))


# -------- multi-gt --------


@pytest.mark.unit
def test_multi_gt_partial_recall():
    m = compute_metrics(_trace(["x", "g1", "y", "z", "g2"]), _qa("f-5", ["g1", "g2"]))
    assert m["recall_at_1"] == 0.0
    assert m["recall_at_3"] == pytest.approx(0.5)
    assert m["recall_at_5"] == 1.0
    assert m["mrr"] == pytest.approx(0.5)


@pytest.mark.unit
def test_multi_gt_perfect_when_both_at_top():
    m = compute_metrics(_trace(["g1", "g2", "x", "y"]), _qa("f-5", ["g1", "g2"]))
    assert m["recall_at_1"] == pytest.approx(0.5)
    assert m["recall_at_3"] == 1.0
    assert m["ndcg_at_10"] == pytest.approx(1.0)


@pytest.mark.unit
def test_recall_at_10_when_only_5_retrieved():
    m = compute_metrics(_trace(["a", "b", "c", "target", "d"]), _qa("f-1", ["target"]))
    assert m["recall_at_5"] == 1.0
    assert m["recall_at_10"] == 1.0


# -------- kind filter --------


@pytest.mark.unit
def test_non_chunk_items_excluded_from_ranking():
    """target sits at rank-3 in trace but is the FIRST chunk → MRR = 1.0."""
    items = [
        {"chunk_id": "ent-1", "rank": 1, "score": 0.95, "kind": "entity"},
        {"chunk_id": "ent-2", "rank": 2, "score": 0.94, "kind": "entity"},
        {"chunk_id": "target", "rank": 3, "score": 0.93, "kind": "chunk"},
        {"chunk_id": "com-1", "rank": 4, "score": 0.92, "kind": "community"},
    ]
    m = compute_metrics({"retrieved": items}, _qa("f-1", ["target"]))
    assert m["recall_at_1"] == 1.0
    assert m["mrr"] == 1.0


@pytest.mark.unit
def test_missing_kind_treated_as_chunk():
    items = [
        {"chunk_id": "target", "rank": 1, "score": 0.9},  # no kind
        {"chunk_id": "other", "rank": 2, "score": 0.8},
    ]
    m = compute_metrics({"retrieved": items}, _qa("f-1", ["target"]))
    assert m["recall_at_1"] == 1.0
    assert m["mrr"] == 1.0


# -------- edges --------


@pytest.mark.unit
def test_empty_trace_with_real_gt_is_no_hit_not_refusal():
    m = compute_metrics(_trace([]), _qa("f-1", ["target"]))
    assert m["refusal_correct"] is False
    assert m["recall_at_1"] == 0.0
    assert m["mrr"] == 0.0
