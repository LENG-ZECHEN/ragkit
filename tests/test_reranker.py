"""Tests for the rerank score back-fill (alignment by stable id, not text).

The reranker is mocked end-to-end: no DashScope call is made. The fake stands in
for ``DashScopeRerank`` and returns the same node objects the real model would,
so ``rerank_id`` (input position) survives reordering/omission exactly as it does
in production.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from llama_index.core.schema import NodeWithScore

import ragkit.core.reranker as reranker_mod
from ragkit.core.reranker import rerank_scores


class _FakeReranker:
    """Stand-in for DashScopeRerank driven by a per-test behavior callable.

    The callable receives the input ``list[NodeWithScore]`` and returns the
    (possibly reordered / subset) list the real model would.
    """

    def __init__(self, behavior: Callable[[list[NodeWithScore]], list[NodeWithScore]]):
        self._behavior = behavior

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        return self._behavior(nodes)


def _install(monkeypatch: pytest.MonkeyPatch, behavior: Callable) -> None:
    monkeypatch.setattr(
        reranker_mod, "DashScopeRerank", lambda **kwargs: _FakeReranker(behavior)
    )


def _rescore(
    nodes: list[NodeWithScore], scores_by_index: dict[int, float]
) -> list[NodeWithScore]:
    """Rebuild NodeWithScore (same node objects) scored by input position."""
    return [
        NodeWithScore(node=nws.node, score=scores_by_index[nws.node.metadata["rerank_id"]])
        for nws in nodes
        if nws.node.metadata["rerank_id"] in scores_by_index
    ]


@pytest.mark.unit
def test_duplicate_texts_get_independent_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    # index 0 and 2 are identical text; they must not collide.
    texts = ["alpha", "beta", "alpha"]
    target = {0: 0.9, 1: 0.5, 2: 0.1}
    _install(monkeypatch, lambda nodes: _rescore(nodes, target))

    scores = rerank_scores("q", texts)

    assert scores.tolist() == pytest.approx([0.9, 0.5, 0.1])
    # The two identical chunks keep their own scores (text-keyed code would
    # collapse both to a single value).
    assert scores[0] != scores[2]


@pytest.mark.unit
def test_shuffled_return_order_realigns(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = ["a", "b", "c", "d"]
    target = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    # Reranker returns nodes in reversed order.
    _install(monkeypatch, lambda nodes: list(reversed(_rescore(nodes, target))))

    scores = rerank_scores("q", texts)

    # Scores come back in INPUT order despite the reranker shuffling them.
    assert scores.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


@pytest.mark.unit
def test_missing_result_warns_and_does_not_silently_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ["a", "b", "c"]
    # The reranker omits index 1 entirely.
    _install(monkeypatch, lambda nodes: _rescore(nodes, {0: 0.8, 2: 0.4}))

    warnings: list[tuple] = []
    monkeypatch.setattr(
        reranker_mod.logger, "warning", lambda *a, **k: warnings.append((a, k))
    )

    scores = rerank_scores("q", texts)

    assert scores[0] == pytest.approx(0.8)
    assert scores[2] == pytest.approx(0.4)
    # The missing input is NOT silently 0.0 — it falls back to the lowest
    # observed score, and a warning is emitted.
    assert scores[1] != 0.0
    assert scores[1] == pytest.approx(0.4)
    assert warnings, "a warning must be emitted when a score is missing"


@pytest.mark.unit
def test_empty_input_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # No reranker call should happen for empty input.
    _install(monkeypatch, lambda nodes: [])
    assert rerank_scores("q", []).tolist() == []
