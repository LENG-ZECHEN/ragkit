"""LLM-based description consolidator — covers threshold, batching, failure
tolerance, max_calls cap, and bypass-merge semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.graph.description_merger import (
    CONSOLIDATION_MIN_CHUNKS,
    CONSOLIDATION_THRESHOLD_CHARS,
    ConsolidationResult,
    consolidate_all,
    consolidate_entity_description,
    consolidate_relation_description,
)
from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Entity, Relation


def _store(tmp_path: Path) -> NetworkXGraphStore:
    return NetworkXGraphStore(path=tmp_path / "g.json")


def _long_desc(n: int) -> str:
    """Build a description longer than the threshold (n controls extra padding)."""
    base = "重要事实。" * 60  # ~300 chars
    return base + ("X" * n)


# ----- single-item consolidation ------------------------------------------


def test_consolidate_entity_description_uses_llm(fake_openai):
    """When called, it goes through the OpenAI chat client."""
    fake_openai.chat_script = [("content", "Consolidated description.")]
    entity = Entity(name="qwen", type="model", description="orig long description")
    out = consolidate_entity_description(entity)
    assert out == "Consolidated description."
    assert any(c["kind"] == "chat" for c in fake_openai.calls)


def test_consolidate_entity_returns_none_on_llm_failure(fake_openai, monkeypatch):
    """An LLM error must NOT propagate — return None so caller can skip."""
    def boom(**kw):
        raise RuntimeError("API down")

    monkeypatch.setattr(fake_openai.chat.completions, "create", boom)
    entity = Entity(name="qwen", type="model", description=_long_desc(0))
    assert consolidate_entity_description(entity) is None


def test_consolidate_entity_returns_none_on_empty_llm_output(fake_openai):
    """LLM returned blank → return None, don't overwrite with empty string."""
    fake_openai.chat_script = [("content", "   ")]
    entity = Entity(name="qwen", type="model", description="orig")
    assert consolidate_entity_description(entity) is None


def test_consolidate_relation_description(fake_openai):
    fake_openai.chat_script = [("content", "Consolidated relation.")]
    relation = Relation(source="a", target="b", description="orig")
    assert consolidate_relation_description(relation) == "Consolidated relation."


# ----- threshold gating ---------------------------------------------------


def test_consolidate_all_skips_entities_below_char_threshold(tmp_path, fake_openai):
    """Short descriptions don't trigger consolidation, no matter how many chunks."""
    store = _store(tmp_path)
    store.upsert_entity(Entity(
        name="x",
        type="t",
        description="short",  # well below 250 chars
        source_chunks=["c1", "c2", "c3", "c4", "c5"],  # well above 3 chunks
    ))

    result = consolidate_all(store)

    assert result.total_calls == 0
    assert result.entities_processed == set()


def test_consolidate_all_skips_entities_below_chunk_threshold(tmp_path, fake_openai):
    """Long descriptions from too few chunks (1-2 mentions) are probably
    already focused — don't waste LLM calls on them."""
    store = _store(tmp_path)
    store.upsert_entity(Entity(
        name="x",
        type="t",
        description=_long_desc(0),  # over 250 chars
        source_chunks=["c1"],  # only 1 chunk → below threshold
    ))

    result = consolidate_all(store)

    assert result.total_calls == 0


def test_consolidate_all_processes_qualifying_entities(tmp_path, fake_openai):
    """Both thresholds met → entity gets consolidated."""
    fake_openai.chat_script = [("content", "Concise summary.")]
    store = _store(tmp_path)
    store.upsert_entity(Entity(
        name="qwen",
        type="model",
        description=_long_desc(0),
        source_chunks=["c1", "c2", "c3", "c4"],
    ))

    result = consolidate_all(store)

    assert result.total_calls == 1
    assert "qwen" in result.entities_processed
    e = store.get_entity("qwen")
    assert e.description == "Concise summary."


def test_consolidate_all_processes_qualifying_relations(tmp_path, fake_openai):
    fake_openai.chat_script = [("content", "Concise relation.")]
    store = _store(tmp_path)
    # Endpoints must exist so the edge can hold a long description.
    store.upsert_relation(Relation(
        source="a",
        target="b",
        description=_long_desc(0),
        source_chunks=["c1", "c2", "c3", "c4"],
    ))

    result = consolidate_all(store)

    assert result.total_calls == 1
    assert ("a", "b") in result.relations_processed
    rels = list(store.all_relations())
    assert rels[0].description == "Concise relation."


# ----- max_calls cap ------------------------------------------------------


def test_consolidate_all_respects_max_calls_cap(tmp_path, fake_openai):
    """When more candidates than max_calls, stop early."""
    fake_openai.chat_script = [("content", "Consolidated.")]
    store = _store(tmp_path)
    for i in range(10):
        store.upsert_entity(Entity(
            name=f"e{i}",
            type="t",
            description=_long_desc(i),
            source_chunks=[f"c{j}" for j in range(5)],
        ))

    result = consolidate_all(store, max_calls=3)

    assert result.total_calls == 3
    # Cap honored — only first 3 (the largest descriptions) processed.
    assert len(result.entities_processed) == 3


def test_consolidate_all_zero_max_calls_processes_nothing(tmp_path, fake_openai):
    fake_openai.chat_script = [("content", "X")]
    store = _store(tmp_path)
    store.upsert_entity(Entity(
        name="x",
        type="t",
        description=_long_desc(0),
        source_chunks=["c1", "c2", "c3", "c4"],
    ))

    result = consolidate_all(store, max_calls=0)

    assert result.total_calls == 0


# ----- failure handling ---------------------------------------------------


def test_consolidate_all_continues_on_per_item_failure(tmp_path, fake_openai, monkeypatch):
    """One LLM failure must not abort the whole sweep."""
    call_count = {"n": 0}
    real_create = fake_openai.chat.completions.create

    def flaky(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient")
        return real_create(**kw)

    monkeypatch.setattr(fake_openai.chat.completions, "create", flaky)
    fake_openai.chat_script = [("content", "Good summary.")]

    store = _store(tmp_path)
    for i in range(3):
        store.upsert_entity(Entity(
            name=f"e{i}",
            type="t",
            description=_long_desc(i),
            source_chunks=[f"c{j}" for j in range(5)],
        ))

    result = consolidate_all(store)

    # First entity failed; remaining 2 succeeded.
    assert result.total_calls == 3
    assert result.failures == 1
    assert len(result.entities_processed) == 2


# ----- bypass-merge semantics --------------------------------------------


def test_consolidate_replaces_not_concatenates(tmp_path, fake_openai):
    """If consolidator went through upsert_entity, merge() would concatenate
    'long original' + ' ' + 'short summary' — exactly what we want to avoid.
    This test confirms the consolidator uses replace_*_description."""
    fake_openai.chat_script = [("content", "Short summary")]
    store = _store(tmp_path)
    long_original = _long_desc(0)
    store.upsert_entity(Entity(
        name="x",
        type="t",
        description=long_original,
        source_chunks=["c1", "c2", "c3", "c4", "c5"],
    ))

    consolidate_all(store)

    e = store.get_entity("x")
    assert e.description == "Short summary"
    assert long_original not in e.description  # Old text fully replaced


# ----- progress callback --------------------------------------------------


def test_consolidate_all_fires_progress_callback(tmp_path, fake_openai):
    fake_openai.chat_script = [("content", "summary")]
    store = _store(tmp_path)
    for i in range(2):
        store.upsert_entity(Entity(
            name=f"e{i}",
            type="t",
            description=_long_desc(0),
            source_chunks=["c1", "c2", "c3", "c4"],
        ))

    stages: list[str] = []

    def cb(stage, current, total):
        stages.append(stage)
        assert current <= total

    consolidate_all(store, progress_cb=cb)

    assert "consolidating" in stages
    assert len(stages) >= 2


# ----- thresholds are sane ------------------------------------------------


def test_threshold_constants_have_safe_buffer():
    """Regression guard: the gap between trigger and target must be wide
    enough that LLM overshoot doesn't immediately re-trigger consolidation."""
    assert CONSOLIDATION_THRESHOLD_CHARS > 200
    assert CONSOLIDATION_MIN_CHUNKS >= 2


# ----- empty graph --------------------------------------------------------


def test_consolidate_all_empty_graph_returns_empty_result(tmp_path, fake_openai):
    store = _store(tmp_path)
    result = consolidate_all(store)
    assert result == ConsolidationResult()
    assert result.total_calls == 0
