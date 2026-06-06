"""Tests for global_search.py — Microsoft GraphRAG's Map-Reduce pipeline.

Covers helpers (shuffle, batch-by-token, parse) and the full
run_global_search flow with mocked LLM.
"""

from __future__ import annotations

import json

import pytest

from ragkit.core.graph.global_search import (
    DEFAULT_FINAL_TOP_K,
    MAP_BATCH_TOKEN_BUDGET,
    RATING_THRESHOLD,
    RatedPoint,
    _batch_by_token_count,
    _estimate_tokens,
    _map_rate_batch,
    _parse_map_response,
    _reduce_rated_points,
    _shuffle_with_seed,
    _strip_code_fence,
    run_global_search,
)


def _report(cid: int, text: str = "report text") -> dict:
    """Build a minimal community report _source dict."""
    return {
        "community_id_int": cid,
        "community_level_int": 0,
        "content_with_weight": text,
    }


# ----- Shuffle and token estimation ---------------------------------------


def test_shuffle_is_deterministic_with_same_seed():
    items = list(range(20))
    a = _shuffle_with_seed(items, seed=42)
    b = _shuffle_with_seed(items, seed=42)
    assert a == b
    # And actually shuffled, not just returning the original.
    assert a != items


def test_shuffle_does_not_mutate_input():
    items = [1, 2, 3, 4, 5]
    _shuffle_with_seed(items)
    assert items == [1, 2, 3, 4, 5]


def test_estimate_tokens_handles_chinese_and_english():
    """Chinese chars cost roughly 2× English. Just check basic ranges."""
    chinese = "中文测试" * 10  # 40 chars
    english = "abcd" * 10      # 40 chars
    assert _estimate_tokens(chinese) > _estimate_tokens(english)
    assert _estimate_tokens("") == 1  # baseline +1


# ----- Token-budget batching ----------------------------------------------


def test_batch_by_token_count_packs_small_reports_together():
    """Many tiny reports → one batch."""
    reports = [_report(i, "x" * 10) for i in range(5)]
    batches = _batch_by_token_count(reports, max_tokens=10_000)
    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_batch_by_token_count_splits_when_budget_exceeded():
    """Reports exceeding budget end up in separate batches."""
    big_text = "x" * 4000  # ~1000 tokens by our estimate
    reports = [_report(i, big_text) for i in range(4)]
    batches = _batch_by_token_count(reports, max_tokens=1500)
    assert len(batches) >= 2


def test_batch_by_token_count_keeps_oversized_singletons():
    """A single report that itself exceeds budget still gets its own batch."""
    huge_text = "x" * 100_000
    reports = [_report(0, huge_text)]
    batches = _batch_by_token_count(reports, max_tokens=100)
    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_batch_by_token_count_empty_input():
    assert _batch_by_token_count([]) == []


# ----- Code-fence stripping (LLM output cleanup) --------------------------


def test_strip_code_fence_handles_json_fence():
    assert _strip_code_fence('```json\n{"x": 1}\n```') == '{"x": 1}'


def test_strip_code_fence_plain_passthrough():
    assert _strip_code_fence('{"x": 1}') == '{"x": 1}'


# ----- _parse_map_response ------------------------------------------------


def test_parse_map_response_extracts_rated_points():
    raw = json.dumps({
        "points": [
            {"point": "P1", "rating": 90},
            {"point": "P2", "rating": 50, "source": "Community 3"},
        ]
    })
    out = _parse_map_response(raw)
    assert len(out) == 2
    assert out[0].point == "P1" and out[0].rating == 90
    assert out[1].source == "Community 3"


def test_parse_map_response_drops_empty_points():
    """LLM sometimes returns blank points; drop them."""
    raw = json.dumps({
        "points": [
            {"point": "Good", "rating": 80},
            {"point": "", "rating": 90},  # empty point → drop
            {"point": "   ", "rating": 90},  # whitespace → drop
        ]
    })
    out = _parse_map_response(raw)
    assert len(out) == 1
    assert out[0].point == "Good"


def test_parse_map_response_clamps_rating():
    """Ratings outside [0, 100] are clamped to range."""
    raw = json.dumps({
        "points": [
            {"point": "A", "rating": 999},  # clamped to 100
            {"point": "B", "rating": -3},   # clamped to 0
            {"point": "C", "rating": "high"},  # non-numeric → 0
        ]
    })
    out = _parse_map_response(raw)
    assert out[0].rating == 100
    assert out[1].rating == 0
    assert out[2].rating == 0


def test_parse_map_response_handles_bad_json():
    """Malformed JSON must not crash — return empty list."""
    assert _parse_map_response("not even close to json") == []


def test_parse_map_response_handles_code_fence():
    raw = '```json\n' + json.dumps({"points": [{"point": "P", "rating": 70}]}) + '\n```'
    out = _parse_map_response(raw)
    assert len(out) == 1 and out[0].point == "P"


# ----- _map_rate_batch (single LLM call) ----------------------------------


def test_map_rate_batch_calls_llm_with_question_and_reports(fake_openai):
    fake_openai.chat_script = [("content", json.dumps({
        "points": [{"point": "ANSWER", "rating": 80}]
    }))]
    batch = [_report(0, "report content"), _report(1, "another")]

    out = _map_rate_batch("user question", batch)

    assert len(out) == 1
    assert out[0].point == "ANSWER"


def test_map_rate_batch_returns_empty_on_llm_failure(fake_openai, monkeypatch):
    """An LLM error must not abort the whole global search — return [] for
    this batch and let other batches contribute."""
    def boom(**kw):
        raise RuntimeError("API down")

    monkeypatch.setattr(fake_openai.chat.completions, "create", boom)

    out = _map_rate_batch("q", [_report(0, "x")])
    assert out == []


# ----- _reduce_rated_points -----------------------------------------------


def test_reduce_filters_below_threshold():
    points = [
        RatedPoint(point="A", rating=90),
        RatedPoint(point="B", rating=40),  # below default 50
        RatedPoint(point="C", rating=70),
    ]
    out = _reduce_rated_points(points)
    kept_texts = {p.point for p in out}
    assert kept_texts == {"A", "C"}


def test_reduce_sorts_by_rating_desc():
    points = [
        RatedPoint(point="low", rating=50),
        RatedPoint(point="high", rating=100),
        RatedPoint(point="mid", rating=70),
    ]
    out = _reduce_rated_points(points)
    assert [p.point for p in out] == ["high", "mid", "low"]


def test_reduce_respects_top_k():
    points = [RatedPoint(point=f"p{i}", rating=100) for i in range(50)]
    out = _reduce_rated_points(points, top_k=5)
    assert len(out) == 5


def test_reduce_empty_input():
    assert _reduce_rated_points([]) == []


# ----- run_global_search end-to-end ---------------------------------------


def test_run_global_search_full_pipeline(fake_openai):
    """E2E: shuffle → batch → map (mocked LLM) → reduce."""
    fake_openai.chat_script = [("content", json.dumps({
        "points": [{"point": "ANSWER_POINT", "rating": 90}]
    }))]
    reports = [_report(i, "report text") for i in range(3)]

    out = run_global_search("what are the themes?", reports)

    assert len(out) >= 1
    assert any(p.point == "ANSWER_POINT" for p in out)


def test_run_global_search_empty_input():
    assert run_global_search("q", []) == []


def test_run_global_search_drops_all_below_threshold(fake_openai):
    """When all LLM ratings are below threshold, final list is empty."""
    fake_openai.chat_script = [("content", json.dumps({
        "points": [{"point": "irrelevant", "rating": 30}]
    }))]
    reports = [_report(0)]
    out = run_global_search("q", reports, rating_threshold=50)
    assert out == []


def test_run_global_search_constants_are_reasonable():
    """Regression guard on tunables."""
    assert 0 < RATING_THRESHOLD <= 100
    assert MAP_BATCH_TOKEN_BUDGET >= 500
    assert DEFAULT_FINAL_TOP_K >= 5


# ===========================================================================
# Concurrency tests — verify that Map phase actually runs in parallel
# ===========================================================================


def test_map_phase_runs_concurrently(fake_openai, monkeypatch):
    """Wall-clock time for N slow batches should be ~max_one_batch, NOT N*one.

    Mocks the LLM call to sleep 0.5s. With 3 batches under serial execution
    total time ≈ 1.5s; under concurrent execution (max_workers=5) ≈ 0.5s.
    Assert under 1.0s to give CI headroom (real concurrent should be 0.5-0.6s).
    """
    import time

    def slow_create(**kwargs):
        time.sleep(0.5)  # simulate LLM latency
        mock_msg = type("M", (), {"content": json.dumps({
            "points": [{"point": "p", "rating": 90}]
        })})()
        mock_choice = type("C", (), {"message": mock_msg})()
        return type("R", (), {"choices": [mock_choice]})()

    monkeypatch.setattr(
        fake_openai.chat.completions, "create", slow_create
    )

    # Force 3 separate batches by making each report exceed half the budget.
    big_text = "x" * (MAP_BATCH_TOKEN_BUDGET * 3)  # each ~6000 char ≈ 1500 tok
    reports = [_report(i, big_text) for i in range(3)]

    start = time.monotonic()
    out = run_global_search("test question", reports)
    elapsed = time.monotonic() - start

    # 3 batches × 0.5s serial = 1.5s; concurrent ≈ 0.5s.
    # 1.0s ceiling proves we're concurrent (with CI/macOS scheduler slack).
    assert elapsed < 1.0, (
        f"Map phase took {elapsed:.2f}s — expected ~0.5s under concurrent "
        f"execution (3 batches × 0.5s serial would be 1.5s). "
        f"Concurrency may have regressed to serial."
    )
    assert len(out) >= 1


def test_map_phase_one_batch_failure_does_not_abort_others(fake_openai, monkeypatch):
    """A single batch raising must NOT lose other batches' results.

    This guards the per-future try/except in run_global_search.
    """
    call_count = {"n": 0}

    def flaky_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:  # second batch fails
            raise RuntimeError("simulated transient API error")
        mock_msg = type("M", (), {"content": json.dumps({
            "points": [{"point": f"survived-{call_count['n']}", "rating": 80}]
        })})()
        mock_choice = type("C", (), {"message": mock_msg})()
        return type("R", (), {"choices": [mock_choice]})()

    monkeypatch.setattr(
        fake_openai.chat.completions, "create", flaky_create
    )

    big_text = "x" * (MAP_BATCH_TOKEN_BUDGET * 3)
    reports = [_report(i, big_text) for i in range(3)]

    out = run_global_search("q", reports)

    # 2 surviving batches should each contribute 1 point.
    assert len(out) == 2
    assert all(p.point.startswith("survived-") for p in out)
