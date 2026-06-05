"""Tests for the graph→ES indexing pipeline (task #24).

Covers:
- pure document builders (deterministic IDs, hash fields, correct mapping)
- diff logic (which entities need re-embedding)
- delete+rewrite of community docs
- full pipeline flow + failure thresholds
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragkit.core.graph.es_indexer import (
    EMBED_FAILURE_ABORT_RATIO,
    _build_community_doc,
    _build_entity_doc,
    _community_embed_text,
    _community_doc_id,
    _delete_community_docs,
    _entity_desc_hash,
    _entity_doc_id,
    _entity_embed_text,
    _fetch_existing_entity_hashes,
    _filter_entities_to_embed,
    index_graph_to_es,
)
from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Community, Entity, Finding, Relation


def _store(tmp_path: Path) -> NetworkXGraphStore:
    return NetworkXGraphStore(path=tmp_path / "g.json")


# ============================================================
# Pure helpers — IDs and hashes
# ============================================================


def test_entity_doc_id_is_deterministic():
    """Same name → same id; load-bearing for skip-when-unchanged logic."""
    a = _entity_doc_id("qwen")
    b = _entity_doc_id("qwen")
    assert a == b
    assert a.startswith("ent-")


def test_entity_doc_id_differs_per_name():
    assert _entity_doc_id("qwen") != _entity_doc_id("dashscope")


def test_entity_desc_hash_changes_when_description_changes():
    """Re-embed depends on this — any description change must shift hash."""
    e1 = Entity(name="x", type="t", description="v1")
    e2 = Entity(name="x", type="t", description="v2")
    assert _entity_desc_hash(e1) != _entity_desc_hash(e2)


def test_entity_desc_hash_unchanged_when_irrelevant_fields_change():
    """source_chunks isn't part of the embed text — hash should ignore it."""
    e1 = Entity(name="x", type="t", description="same", source_chunks=["c1"])
    e2 = Entity(name="x", type="t", description="same", source_chunks=["c1", "c2"])
    assert _entity_desc_hash(e1) == _entity_desc_hash(e2)


def test_community_doc_id_encodes_level_and_id():
    c = Community(id=7, level=1, entity_names=[])
    assert _community_doc_id(c) == "com-1-7"


# ============================================================
# Embed text composition
# ============================================================


def test_entity_embed_text_starts_with_name():
    """Names are strong identifiers — must be at the start of the embed text."""
    e = Entity(name="qwen", type="model", description="alibaba's llm")
    out = _entity_embed_text(e)
    assert out.startswith("qwen")
    assert "alibaba's llm" in out


def test_community_embed_text_includes_title_summary_findings():
    """Vector should reflect title + summary + finding heads (not full bodies)."""
    c = Community(
        id=0,
        entity_names=["x"],
        title="THE_TITLE",
        summary="THE_SUMMARY",
        findings=[
            Finding(summary="FIND_HEAD", explanation="LONG_EXPLANATION_NOT_IN_EMBED"),
        ],
    )
    text = _community_embed_text(c)
    assert "THE_TITLE" in text
    assert "THE_SUMMARY" in text
    assert "FIND_HEAD" in text
    # Long explanations should NOT pollute the embedding vector
    assert "LONG_EXPLANATION_NOT_IN_EMBED" not in text


def test_community_embed_text_falls_back_to_id_when_empty():
    """A community with no LLM report still needs SOME embed text."""
    c = Community(id=42, entity_names=["x"])
    text = _community_embed_text(c)
    assert "42" in text


# ============================================================
# Document builders
# ============================================================


def test_build_entity_doc_has_required_fields():
    e = Entity(name="qwen", type="model", description="d", source_chunks=["c1", "c2"])
    doc = _build_entity_doc(e, embedding=[0.1, 0.2], kb_name="kb")

    assert doc["id"].startswith("ent-")
    assert doc["type_kwd"] == "entity"
    assert doc["kb_id"] == "kb"
    assert doc["entity_name_kwd"] == "qwen"
    assert doc["entity_type_kwd"] == "model"
    assert doc["source_chunks_kwd"] == ["c1", "c2"]
    assert "desc_hash_kwd" in doc and len(doc["desc_hash_kwd"]) > 0
    # Vector field is dimension-tagged per the ES mapping convention
    assert "q_2_vec" in doc
    assert doc["q_2_vec"] == [0.1, 0.2]


def test_build_entity_doc_handles_missing_type():
    """Default type to 'unknown' if extractor left it empty."""
    e = Entity(name="x", type="", description="d")
    doc = _build_entity_doc(e, embedding=[0.0], kb_name="kb")
    assert doc["entity_type_kwd"] == "unknown"


def test_build_community_doc_has_required_fields():
    c = Community(
        id=3,
        level=1,
        entity_names=["a", "b"],
        title="T",
        summary="S",
        rank=7.5,
        findings=[Finding(summary="fs", explanation="fe")],
    )
    doc = _build_community_doc(c, embedding=[0.1] * 4, kb_name="kb")

    assert doc["id"] == "com-1-3"
    assert doc["type_kwd"] == "community"
    assert doc["kb_id"] == "kb"
    assert doc["community_level_int"] == 1
    assert doc["community_id_int"] == 3
    assert doc["community_rank_flt"] == 7.5
    assert doc["community_entity_names_kwd"] == ["a", "b"]
    assert "q_4_vec" in doc


# ============================================================
# Diff logic — filter_entities_to_embed
# ============================================================


def test_filter_includes_unknown_entities(tmp_path):
    """Entity not in ES at all → must be embedded."""
    store = _store(tmp_path)
    store.upsert_entity(Entity(name="new1", type="t", description="d"))
    store.upsert_entity(Entity(name="new2", type="t", description="d"))

    out = _filter_entities_to_embed(store, existing_hashes={})

    assert {e.name for e in out} == {"new1", "new2"}


def test_filter_skips_unchanged_entities(tmp_path):
    """Entity with matching hash already in ES → skip."""
    store = _store(tmp_path)
    e = Entity(name="qwen", type="t", description="unchanged")
    store.upsert_entity(e)

    # Pre-compute the hash that ES would have stored for this exact entity.
    existing_hashes = {"qwen": _entity_desc_hash(e)}

    out = _filter_entities_to_embed(store, existing_hashes)
    assert out == []


def test_filter_includes_changed_entities(tmp_path):
    """Stored hash differs from current → re-embed."""
    store = _store(tmp_path)
    store.upsert_entity(Entity(name="qwen", type="t", description="new desc"))

    existing_hashes = {"qwen": "stale_hash_value"}

    out = _filter_entities_to_embed(store, existing_hashes)
    assert len(out) == 1 and out[0].name == "qwen"


# ============================================================
# _fetch_existing_entity_hashes — ES query shape
# ============================================================


def test_fetch_existing_hashes_returns_empty_when_no_index():
    fake_raw = MagicMock()
    fake_raw.indices.exists.return_value = False
    assert _fetch_existing_entity_hashes("kb", fake_raw) == {}


def test_fetch_existing_hashes_walks_scroll_pages():
    """Scrolls through pages until hits run out."""
    fake_raw = MagicMock()
    fake_raw.indices.exists.return_value = True
    # First page has 2 entities; second is empty (terminates the loop)
    fake_raw.search.return_value = {
        "_scroll_id": "sid",
        "hits": {"hits": [
            {"_source": {"entity_name_kwd": "a", "desc_hash_kwd": "h_a"}},
            {"_source": {"entity_name_kwd": "b", "desc_hash_kwd": "h_b"}},
        ]},
    }
    fake_raw.scroll.return_value = {
        "_scroll_id": "sid",
        "hits": {"hits": []},
    }

    out = _fetch_existing_entity_hashes("kb", fake_raw)
    assert out == {"a": "h_a", "b": "h_b"}


# ============================================================
# _delete_community_docs
# ============================================================


def test_delete_community_docs_targets_only_community_type():
    """The delete_by_query body must filter type_kwd=community AND kb_id.

    Updated for ISS-015: we now pass modern elasticsearch-py 8.x kwargs
    (query=... directly) instead of the deprecated body={query: ...}.
    """
    fake_raw = MagicMock()
    fake_raw.indices.exists.return_value = True

    _delete_community_docs("kb1", fake_raw)

    call = fake_raw.delete_by_query.call_args
    # Accept both modern kwarg style and legacy body= style.
    query = call.kwargs.get("query") or (call.kwargs.get("body") or {}).get("query")
    assert query is not None, f"No query found in delete_by_query call: {call.kwargs}"
    must = query["bool"]["must"]
    types = [m["term"].get("type_kwd") for m in must if "type_kwd" in m["term"]]
    kbs = [m["term"].get("kb_id") for m in must if "kb_id" in m["term"]]
    assert "community" in types
    assert "kb1" in kbs


def test_delete_community_docs_skips_when_index_absent():
    """Don't call delete_by_query on a non-existent index."""
    fake_raw = MagicMock()
    fake_raw.indices.exists.return_value = False
    _delete_community_docs("kb", fake_raw)
    fake_raw.delete_by_query.assert_not_called()


# ============================================================
# Full pipeline — index_graph_to_es
# ============================================================


def _populated_store_with_communities(tmp_path):
    store = _store(tmp_path)
    store.upsert_entity(Entity(name="qwen", type="model", description="alibaba llm"))
    store.upsert_entity(Entity(name="dashscope", type="platform", description="hosting"))
    store.upsert_relation(Relation(source="qwen", target="dashscope", description="on"))
    store.set_communities([
        Community(
            id=0,
            level=0,
            entity_names=["qwen", "dashscope"],
            title="qwen ecosystem",
            summary="The qwen series and its hosting platform.",
            rank=8.0,
            findings=[Finding(summary="qwen is alibaba's LLM", explanation="...")],
        ),
    ])
    return store


def test_index_graph_to_es_embeds_and_writes_entities(tmp_path, fake_openai, fake_es):
    """First-time indexing of a populated graph: every entity gets embedded."""
    fake_es.es.indices.exists.return_value = False  # no existing index
    fake_es.es.search.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}
    fake_es.es.scroll.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}

    store = _populated_store_with_communities(tmp_path)
    result = index_graph_to_es(store, "kb")

    assert result["entity_embedded"] == 2
    assert result["community_embedded"] == 1
    fake_es.ensure_index.assert_called_with("kb_graph")


def test_index_graph_to_es_skips_unchanged_entities(tmp_path, fake_openai, fake_es, monkeypatch):
    """Second build with no changes: entities are detected as unchanged and skipped."""
    store = _populated_store_with_communities(tmp_path)
    # Build hashes that match what's currently in the store.
    existing = {e.name: _entity_desc_hash(e) for e in store.all_entities()}

    monkeypatch.setattr(
        "ragkit.core.graph.es_indexer._fetch_existing_entity_hashes",
        lambda kb, es: existing,
    )

    result = index_graph_to_es(store, "kb")
    assert result["entity_embedded"] == 0  # all skipped
    # Communities are always refreshed
    assert result["community_embedded"] == 1


def test_index_graph_to_es_aborts_on_high_failure_rate(
    tmp_path, fake_openai, fake_es, monkeypatch
):
    """If most embeddings fail, abort to avoid a half-indexed graph."""
    fake_es.es.indices.exists.return_value = False
    fake_es.es.search.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}
    fake_es.es.scroll.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}

    # Create lots of entities so the failure ratio threshold is meaningful.
    store = _store(tmp_path)
    for i in range(10):
        store.upsert_entity(Entity(name=f"e{i}", type="t", description=f"d{i}"))

    # Force embed_batch to always fail.
    def boom(texts):
        raise RuntimeError("API down")

    monkeypatch.setattr("ragkit.core.graph.es_indexer.embed_batch", boom)

    with pytest.raises(RuntimeError, match="embedding failed"):
        index_graph_to_es(store, "kb")


def test_index_graph_to_es_skips_communities_without_content(tmp_path, fake_openai, fake_es):
    """Empty communities (no title/summary/findings) shouldn't be embedded."""
    fake_es.es.indices.exists.return_value = False
    fake_es.es.search.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}
    fake_es.es.scroll.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}

    store = _store(tmp_path)
    store.upsert_entity(Entity(name="x", type="t", description="d"))
    # One community with no content
    store.set_communities([Community(id=0, level=0, entity_names=["x"])])

    result = index_graph_to_es(store, "kb")
    assert result["community_embedded"] == 0


def test_index_graph_to_es_progress_callback_fires(tmp_path, fake_openai, fake_es):
    fake_es.es.indices.exists.return_value = False
    fake_es.es.search.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}
    fake_es.es.scroll.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}

    store = _populated_store_with_communities(tmp_path)
    stages: list[str] = []

    def cb(stage, current, total):
        stages.append(stage)

    index_graph_to_es(store, "kb", progress_cb=cb)
    assert "embedding_entities" in stages
    assert "embedding_communities" in stages


def test_index_graph_to_es_empty_store_is_noop(tmp_path, fake_openai, fake_es):
    """No entities, no communities — must not crash."""
    fake_es.es.indices.exists.return_value = False
    fake_es.es.search.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}
    fake_es.es.scroll.return_value = {"_scroll_id": "sid", "hits": {"hits": []}}

    store = _store(tmp_path)
    result = index_graph_to_es(store, "kb")
    assert result["entity_embedded"] == 0
    assert result["community_embedded"] == 0


def test_failure_ratio_constant_is_strict_enough():
    """Regression guard — the abort threshold must be tight."""
    assert 0 < EMBED_FAILURE_ABORT_RATIO <= 0.5
