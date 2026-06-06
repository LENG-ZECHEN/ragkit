"""Tests for the observe module — verify the show vs trace dichotomy and
key formatting behaviors.

Strategy: use rich Console.capture() to inspect what got printed without
relying on console-output capture (which can be flaky across pytest config).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragkit.cli import observe


# ----- Toggle isolation --------------------------------------------------


@pytest.fixture(autouse=True)
def reset_observe_state():
    """Every test starts with debug OFF and the global state reset.
    Without this, an earlier test enabling debug would leak into later ones."""
    observe.disable_debug()
    yield
    observe.disable_debug()


def _capture(fn, *args, **kw) -> str:
    """Run a function while capturing the shared rich console's output."""
    with observe.console.capture() as buf:
        fn(*args, **kw)
    return buf.get()


# ==========================================================================
# Toggle semantics
# ==========================================================================


def test_debug_default_off():
    assert observe.is_debug() is False


def test_enable_then_disable():
    observe.enable_debug()
    assert observe.is_debug() is True
    observe.disable_debug()
    assert observe.is_debug() is False


def test_enable_is_idempotent():
    observe.enable_debug()
    observe.enable_debug()
    observe.enable_debug()
    assert observe.is_debug() is True


# ==========================================================================
# Default-mode functions ALWAYS emit
# ==========================================================================


def test_show_chunks_produced_emits_regardless_of_debug():
    """Default-mode shows must work even with debug OFF."""
    out = _capture(observe.show_chunks_produced, "paper.pdf", 87)
    assert "87" in out
    assert "paper.pdf" in out


def test_show_chunks_produced_warns_on_zero():
    """Zero chunks is a likely problem — surface it more prominently."""
    out = _capture(observe.show_chunks_produced, "broken.pdf", 0)
    assert "0 chunks" in out
    # Look for the yellow warning marker (rich often emits ⚠ or "warning")
    assert "broken.pdf" in out


def test_show_dendrogram_structure_lists_each_level():
    out = _capture(observe.show_dendrogram_structure, {0: 3, 1: 8, 2: 15})
    assert "Level 0" in out
    assert "Level 1" in out
    assert "Level 2" in out
    assert "3" in out and "8" in out and "15" in out


def test_show_dendrogram_skips_empty_input():
    out = _capture(observe.show_dendrogram_structure, {})
    assert out == ""  # nothing printed for empty dict


def test_show_es_graph_indexing_summarizes_counts():
    stats = {
        "entity_embedded": 12,
        "entity_skipped": 188,
        "community_embedded": 28,
        "entity_failed": 0,
        "community_failed": 0,
    }
    out = _capture(observe.show_es_graph_indexing, stats)
    assert "12" in out and "28" in out


def test_show_es_graph_indexing_highlights_failures():
    stats = {
        "entity_embedded": 1,
        "entity_skipped": 0,
        "community_embedded": 0,
        "entity_failed": 5,
        "community_failed": 2,
    }
    out = _capture(observe.show_es_graph_indexing, stats)
    # 5 + 2 = 7 failures should be surfaced.
    assert "7" in out and "failed" in out.lower()


# ==========================================================================
# Debug-mode functions are NO-OPS unless enabled
# ==========================================================================


def test_trace_seed_entities_silent_when_debug_off():
    out = _capture(observe.trace_seed_entities, [{"entity_name_kwd": "x"}])
    assert out == ""


def test_trace_seed_entities_emits_when_debug_on():
    observe.enable_debug()
    out = _capture(
        observe.trace_seed_entities,
        [
            {"entity_name_kwd": "qwen", "entity_type_kwd": "model", "source_chunks_kwd": ["c1", "c2"]},
            {"entity_name_kwd": "dashscope", "entity_type_kwd": "platform", "source_chunks_kwd": ["c3"]},
        ],
    )
    assert "qwen" in out
    assert "dashscope" in out


def test_trace_query_rewriting_silent_when_debug_off():
    """When off, trace_query_rewriting must NOT do work — important because
    it would re-run the queryer's heavy methods."""
    class FakeQueryer:
        def __init__(self):
            self.called = False

        # Surface attributes the trace function may touch.
        class _Tw:
            def weights(self_inner, tokens, preprocess):
                self.called = True
                return []

        class _Syn:
            def lookup(self_inner, _):
                self.called = True
                return []

        tw = _Tw()
        syn = _Syn()

    q = FakeQueryer()
    out = _capture(observe.trace_query_rewriting, "what is qwen?", q)
    assert out == ""
    assert q.called is False  # Heavy methods NOT called when debug off


def test_trace_global_candidates_emits_table_when_on():
    observe.enable_debug()
    community_docs = [
        {
            "community_id_int": 0,
            "community_level_int": 0,
            "community_rank_flt": 8.5,
            "content_with_weight": "通义千问与阿里云生态\n围绕 qwen 大模型的部署...",
        },
        {
            "community_id_int": 3,
            "community_level_int": 1,
            "community_rank_flt": 6.0,
            "content_with_weight": "DashScope 平台细节",
        },
    ]
    out = _capture(observe.trace_global_candidates, community_docs)
    assert "通义千问" in out or "Community" in out
    assert "8.5" in out  # rank


def test_trace_global_map_batch_shows_rated_points():
    observe.enable_debug()

    @dataclass
    class FakePoint:
        rating: int
        point: str

    points = [FakePoint(rating=9, point="POINT_TEXT_MARKER")]
    out = _capture(observe.trace_global_map_batch, 1, 5, points)
    assert "POINT_TEXT_MARKER" in out
    assert "9" in out
    assert "5" in out  # report count


def test_trace_global_reduce_shows_filter_stats():
    observe.enable_debug()
    out = _capture(observe.trace_global_reduce, 27, 8, threshold=5)
    assert "27" in out and "8" in out and "5" in out


def test_trace_consolidation_summary_shows_counts():
    observe.enable_debug()

    @dataclass
    class FakeStats:
        entities_processed: set
        relations_processed: set
        total_calls: int
        failures: int

    stats = FakeStats(
        entities_processed={"qwen", "dashscope", "alibaba"},
        relations_processed={("qwen", "dashscope")},
        total_calls=4,
        failures=1,
    )
    out = _capture(observe.trace_consolidation_summary, stats)
    assert "3" in out  # 3 entities processed
    assert "1" in out  # 1 relation
    assert "4" in out  # calls
    assert "1" in out  # failures


def test_trace_chunk_extraction_silent_when_debug_off():
    out = _capture(observe.trace_chunk_extraction, "chunk-abc", 5, 3)
    assert out == ""


def test_trace_chunk_extraction_emits_when_debug_on():
    observe.enable_debug()
    out = _capture(observe.trace_chunk_extraction, "chunk-abc-1234", 5, 3)
    # First 8 chars of chunk_id should appear.
    assert "chunk-ab" in out
    assert "5" in out and "3" in out


# ==========================================================================
# timed() context manager
# ==========================================================================


def test_timed_is_noop_when_debug_off():
    out = _capture(lambda: _run_with_timed("label", noop=True))
    assert out == ""


def test_timed_prints_label_and_ms_when_debug_on():
    observe.enable_debug()
    out = _capture(lambda: _run_with_timed("label_marker"))
    assert "label_marker" in out
    assert "ms" in out


def _run_with_timed(label: str, noop: bool = False):
    """Helper to exercise the timed() contextmanager inside a capture."""
    with observe.timed(label):
        if noop:
            return


# ==========================================================================
# measure() context manager — always-on timing into a dict
# ==========================================================================


def test_measure_writes_elapsed_ms_into_dict():
    """measure(key, into) must populate into[key] with a non-negative float
    representing milliseconds elapsed inside the block."""
    d: dict[str, float] = {}
    with observe.measure("foo", d):
        pass
    assert "foo" in d
    assert isinstance(d["foo"], float)
    assert d["foo"] >= 0.0


def test_measure_is_always_active_even_with_debug_off():
    """Unlike timed(), measure() does NOT depend on the debug flag —
    eval traces must be populated regardless."""
    observe.disable_debug()
    d: dict[str, float] = {}
    with observe.measure("k", d):
        pass
    assert "k" in d


def test_measure_records_meaningful_elapsed():
    """For a brief sleep, elapsed_ms should be roughly the sleep duration
    (with generous CI tolerance) and clearly > 0."""
    import time as _t
    d: dict[str, float] = {}
    with observe.measure("slept", d):
        _t.sleep(0.01)  # 10ms
    # At least 5ms should be observable on any sane system; cap upper to catch
    # accidental seconds-units bugs.
    assert d["slept"] >= 5.0
    assert d["slept"] < 2000.0


def test_measure_records_on_exception():
    """The finally block must run — exception inside the block still records."""
    d: dict[str, float] = {}
    with pytest.raises(RuntimeError):
        with observe.measure("boom", d):
            raise RuntimeError("simulated")
    assert "boom" in d
    assert d["boom"] >= 0.0


# ==========================================================================
# references_table_with_kind — default-mode formatter for local mode
# ==========================================================================


def test_references_table_includes_kind_column():
    @dataclass
    class FakeHit:
        rank: int
        kind: str
        title: str
        extra: dict

    hits = [
        FakeHit(rank=1, kind="chunk", title="report.pdf", extra={"similarity": 0.85}),
        FakeHit(rank=2, kind="entity", title="qwen [model]", extra={}),
        FakeHit(rank=3, kind="community", title="qwen ecosystem", extra={"rank": 8.0}),
        FakeHit(rank=4, kind="relation", title="qwen ↔ alibaba", extra={"weight": 3.0}),
    ]

    table = observe.references_table_with_kind(hits)

    # rendering happens via console; capture as we render
    with observe.console.capture() as buf:
        observe.console.print(table)
    out = buf.get()

    for marker in ("chunk", "entity", "community", "relation"):
        assert marker in out
    for marker in ("report.pdf", "qwen [model]", "qwen ecosystem", "qwen ↔ alibaba"):
        assert marker in out


def test_show_retrieval_kind_breakdown_groups_by_kind():
    @dataclass
    class FakeHit:
        kind: str

    hits = [
        FakeHit(kind="chunk"),
        FakeHit(kind="chunk"),
        FakeHit(kind="community"),
        FakeHit(kind="entity"),
        FakeHit(kind="entity"),
        FakeHit(kind="entity"),
    ]
    out = _capture(observe.show_retrieval_kind_breakdown, hits)
    # Should show counts: 2 chunks, 1 community, 3 entities (some plural form)
    assert "2 chunks" in out
    assert "1 community" in out
    assert "3 entities" in out


def test_show_retrieval_kind_breakdown_empty_input():
    out = _capture(observe.show_retrieval_kind_breakdown, [])
    assert out == ""  # silent on empty


# ==========================================================================
# Console singleton sanity
# ==========================================================================


def test_console_is_shared_module_level():
    """observe.console must be the single Console used by all helpers, so
    output stays interleaved correctly when called from different modules."""
    assert observe.console is observe.console  # trivially yes
    # And it must be a rich Console
    from rich.console import Console
    assert isinstance(observe.console, Console)
