"""Graph retriever — three modes (local / global / hybrid)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragkit.core.graph.retriever import (
    _find_mentioned_entities,
    graph_hits_to_chunks,
    retrieve_global,
    retrieve_hybrid,
    retrieve_local,
)
from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Community, Entity, Relation
from ragkit.core.retriever import RetrievedChunk


def _make_populated_store(tmp_path: Path) -> NetworkXGraphStore:
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    store.upsert_entity(Entity(
        name="qwen", type="model",
        description="A large language model by Alibaba.",
        source_chunks=["c1", "c2"],
    ))
    store.upsert_entity(Entity(
        name="dashscope", type="platform",
        description="Alibaba's LLM hosting platform.",
        source_chunks=["c1"],
    ))
    store.upsert_entity(Entity(
        name="alibaba", type="organization",
        description="Chinese tech company.",
        source_chunks=["c1", "c3"],
    ))
    store.upsert_relation(Relation(
        source="qwen", target="dashscope",
        description="hosted on", source_chunks=["c1"],
    ))
    store.upsert_relation(Relation(
        source="dashscope", target="alibaba",
        description="operated by", source_chunks=["c1"],
    ))
    return store


# ----- entity matching --------------------------------------------------


def test_find_mentioned_entities_matches_substrings(tmp_path):
    """Substring match handles 'qwen' inside '介绍一下 qwen 模型'."""
    store = _make_populated_store(tmp_path)
    found = _find_mentioned_entities("介绍一下 qwen 模型", store)
    names = {e.name for e in found}
    assert "qwen" in names


def test_find_mentioned_entities_skips_super_short_names(tmp_path):
    """A 1-char entity like 'X' would match almost any question — skip."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    store.upsert_entity(Entity(name="x", type="t"))
    store.upsert_entity(Entity(name="qwen", type="model"))
    found = _find_mentioned_entities("tell me about qwen", store)
    names = {e.name for e in found}
    assert "x" not in names
    assert "qwen" in names


# ----- local retrieval --------------------------------------------------


def test_retrieve_local_returns_seed_and_neighbors(tmp_path):
    """Local query: question mentions qwen → return qwen + connected entities."""
    store = _make_populated_store(tmp_path)
    hits = retrieve_local("what is qwen?", kb_name="t", depth=1, store=store)

    titles = {h.title for h in hits}
    assert any("qwen" in t for t in titles)
    # Within 1 hop we should also see dashscope
    assert any("dashscope" in t for t in titles)


def test_retrieve_local_empty_when_no_match(tmp_path):
    """A question mentioning no known entities returns empty — not an error."""
    store = _make_populated_store(tmp_path)
    hits = retrieve_local("what is the meaning of life", kb_name="t", store=store)
    assert hits == []


def test_retrieve_local_includes_relations_in_content(tmp_path):
    """Local hits must carry the relation context so the LLM can use it."""
    store = _make_populated_store(tmp_path)
    hits = retrieve_local("qwen", kb_name="t", store=store)
    qwen_hit = next(h for h in hits if "qwen" in h.title)
    assert "dashscope" in qwen_hit.content.lower() or "关系" in qwen_hit.content


# ----- global retrieval -------------------------------------------------


def test_retrieve_global_returns_top_summaries(tmp_path):
    store = _make_populated_store(tmp_path)
    store.set_communities([
        Community(id=0, entity_names=["qwen", "dashscope"], summary="LLM platform group."),
        Community(id=1, entity_names=["alibaba"], summary="Parent company."),
    ])
    hits = retrieve_global("tell me about the LLM platform", kb_name="t", top_k=5, store=store)

    assert len(hits) == 2
    # Summary that overlaps with the query should rank first.
    assert "LLM" in hits[0].content


def test_retrieve_global_empty_when_no_summaries(tmp_path):
    """No summaries (graph built without --summarize) → empty list, not crash."""
    store = _make_populated_store(tmp_path)
    # No communities set.
    hits = retrieve_global("anything", kb_name="t", store=store)
    assert hits == []


# ----- hybrid retrieval -------------------------------------------------


def test_retrieve_hybrid_combines_vector_and_local(tmp_path, monkeypatch):
    """Hybrid = vector chunks + local graph hits, deduped."""
    store = _make_populated_store(tmp_path)

    # Mock vector retrieval to return two chunks.
    def fake_vector_retrieve(question, **kwargs):
        return [
            RetrievedChunk(rank=1, document_id="d1", document_name="paper.pdf",
                           content="Qwen is an LLM family.", similarity=0.9,
                           vector_similarity=0.9, term_similarity=0.9),
            RetrievedChunk(rank=2, document_id="d2", document_name="memo.docx",
                           content="DashScope offers API access.", similarity=0.8,
                           vector_similarity=0.8, term_similarity=0.8),
        ]
    monkeypatch.setattr("ragkit.core.graph.retriever.vector_retrieve", fake_vector_retrieve)

    hits = retrieve_hybrid("about qwen", kb_name="t", top_k=3, store=store)

    assert len(hits) > 0
    kinds = {h.kind for h in hits}
    # We expect both kinds in a hybrid result.
    assert "chunk" in kinds
    assert "entity" in kinds


def test_retrieve_hybrid_survives_vector_failure(tmp_path, monkeypatch):
    """If ES is down, hybrid should still return graph results, not crash.
    Vector failure is logged at ERROR level so it doesn't go unnoticed."""
    store = _make_populated_store(tmp_path)

    def broken_vector(*args, **kwargs):
        raise RuntimeError("ES connection refused")

    monkeypatch.setattr("ragkit.core.graph.retriever.vector_retrieve", broken_vector)

    hits = retrieve_hybrid("about qwen", kb_name="t", store=store)

    # No chunk-kind hits because vector failed — caller can detect this.
    assert all(h.kind == "entity" for h in hits)
    assert not any(h.kind == "chunk" for h in hits)


def test_retrieve_hybrid_both_empty_returns_empty(tmp_path, monkeypatch):
    """When both vector AND graph return nothing, hybrid returns []."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")  # empty graph

    def empty_vector(*args, **kwargs):
        return []
    monkeypatch.setattr("ragkit.core.graph.retriever.vector_retrieve", empty_vector)

    hits = retrieve_hybrid("nothing matches", kb_name="t", store=store)
    assert hits == []


def test_retriever_rejects_empty_inputs(tmp_path):
    """All three retrieval modes must validate kb_name and question."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    for fn in (retrieve_local, retrieve_global, retrieve_hybrid):
        with pytest.raises(ValueError):
            fn("", kb_name="kb", store=store)
        with pytest.raises(ValueError):
            fn("q", kb_name="", store=store)


def test_retriever_local_rejects_unbounded_depth(tmp_path):
    """Catch the obvious foot-gun of `depth=1000`."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    with pytest.raises(ValueError, match="depth"):
        retrieve_local("q", kb_name="kb", depth=1000, store=store)


def test_hybrid_dedupes_identical_content(tmp_path, monkeypatch):
    """If a vector chunk and a graph hit have identical content, drop one."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    store.upsert_entity(Entity(name="qwen", type="model", description="The same content here."))

    def fake_vector_retrieve(question, **kwargs):
        return [
            RetrievedChunk(rank=1, document_id="d", document_name="doc",
                           content="The same content here.", similarity=0.9,
                           vector_similarity=0.9, term_similarity=0.9),
        ]
    monkeypatch.setattr("ragkit.core.graph.retriever.vector_retrieve", fake_vector_retrieve)

    hits = retrieve_hybrid("qwen", kb_name="t", store=store)
    contents = [h.content for h in hits]
    # Same content prefix shouldn't appear twice.
    assert len(set(c[:200] for c in contents)) == len(contents)


# ----- conversion -------------------------------------------------------


def test_graph_hits_to_chunks_preserves_order(tmp_path):
    """Conversion mustn't reorder — the generator's citations use rank."""
    from ragkit.core.graph.retriever import GraphHit
    hits = [
        GraphHit(rank=1, kind="entity", title="A", content="a", extra={}),
        GraphHit(rank=2, kind="community", title="B", content="b", extra={}),
        GraphHit(rank=3, kind="chunk", title="C", content="c", extra={"document_id": "doc-c", "similarity": 0.7}),
    ]
    chunks = graph_hits_to_chunks(hits)
    assert [c.rank for c in chunks] == [1, 2, 3]
    assert chunks[2].similarity == 0.7
