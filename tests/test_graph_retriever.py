"""Graph retriever (Microsoft-GraphRAG style) — task #25.

Covers retrieve_local (4-stream multi-source) and retrieve_global
(Map-Reduce). Hybrid mode was removed in task #25.

External calls (ES, embed_one, LLM) are mocked at the searcher /
global_search boundary so the tests run without DashScope or ES.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ragkit.core.graph.global_search import RatedPoint
from ragkit.core.graph.retriever import (
    GraphHit,
    graph_hits_to_chunks,
    retrieve_global,
    retrieve_local,
)
from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Community, Entity, Finding, Relation


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------


def _seed_entity_doc(name: str, source_chunks: list[str] | None = None) -> dict:
    """ES _source dict mimicking what search_entities_by_vector returns."""
    return {
        "entity_name_kwd": name,
        "entity_type_kwd": "model",
        "source_chunks_kwd": source_chunks or ["c1"],
        "content_with_weight": f"{name}: desc",
    }


def _community_doc(cid: int, level: int, entity_names: list[str], summary: str = "") -> dict:
    return {
        "community_id_int": cid,
        "community_level_int": level,
        "community_rank_flt": 8.0,
        "community_entity_names_kwd": entity_names,
        "content_with_weight": summary or f"Community {cid} summary",
    }


def _populated_store(tmp_path: Path) -> NetworkXGraphStore:
    """Local NetworkX graph with a few entities and relations for the
    neighbor / relation streams to walk."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    store.upsert_entity(Entity(name="qwen", type="model", description="阿里 LLM",
                               source_chunks=["c1", "c2"]))
    store.upsert_entity(Entity(name="dashscope", type="platform", description="平台",
                               source_chunks=["c1"]))
    store.upsert_entity(Entity(name="alibaba", type="organization", description="公司",
                               source_chunks=["c2"]))
    store.upsert_relation(Relation(source="qwen", target="dashscope",
                                   description="部署在", weight=3.0))
    store.upsert_relation(Relation(source="dashscope", target="alibaba",
                                   description="隶属于", weight=2.0))
    return store


# ==========================================================================
# Input validation (applies to both retrieve_local and retrieve_global)
# ==========================================================================


def test_retriever_rejects_empty_question(tmp_path):
    store = _populated_store(tmp_path)
    for fn in (retrieve_local, retrieve_global):
        with pytest.raises(ValueError, match="non-empty"):
            fn("", kb_name="kb", store=store)
        with pytest.raises(ValueError, match="non-empty"):
            fn("   ", kb_name="kb", store=store)


def test_retriever_rejects_empty_kb_name(tmp_path):
    store = _populated_store(tmp_path)
    for fn in (retrieve_local, retrieve_global):
        with pytest.raises(ValueError, match="non-empty"):
            fn("q", kb_name="", store=store)


# ==========================================================================
# retrieve_local — Multi-source / 4-stream
# ==========================================================================


def test_local_returns_empty_when_no_seed_entities(tmp_path, monkeypatch):
    """No entity vector search hits → empty result, not crash."""
    store = _populated_store(tmp_path)
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_entities_by_vector",
        lambda kb, q, top_k: [],
    )
    assert retrieve_local("anything", kb_name="kb", store=store) == []


def test_local_collects_chunks_from_seed_source_chunks(tmp_path, monkeypatch):
    """Stream 1 (text units): seed entities' source_chunks → mget {kb}."""
    store = _populated_store(tmp_path)

    # Seed entity references chunks c1, c2.
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_entities_by_vector",
        lambda kb, q, top_k: [_seed_entity_doc("qwen", source_chunks=["c1", "c2"])],
    )
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.fetch_chunks_by_ids",
        lambda kb, ids: [
            {"_id": "c1", "doc_id": "d1", "docnm_kwd": "report.pdf",
             "content_with_weight": "TEXT_FROM_C1"},
            {"_id": "c2", "doc_id": "d1", "docnm_kwd": "report.pdf",
             "content_with_weight": "TEXT_FROM_C2"},
        ],
    )
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_entity_names",
        lambda kb, names, top_k: [],
    )
    hits = retrieve_local("about qwen", kb_name="kb", store=store)

    chunk_hits = [h for h in hits if h.kind == "chunk"]
    assert len(chunk_hits) == 2
    contents = {h.content for h in chunk_hits}
    assert contents == {"TEXT_FROM_C1", "TEXT_FROM_C2"}


def test_local_collects_communities_containing_seed(tmp_path, monkeypatch):
    """Stream 2 (community reports): search_communities_by_entity_names."""
    store = _populated_store(tmp_path)
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_entities_by_vector",
        lambda kb, q, top_k: [_seed_entity_doc("qwen")],
    )
    monkeypatch.setattr("ragkit.core.graph.retriever.fetch_chunks_by_ids", lambda kb, ids: [])
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_entity_names",
        lambda kb, names, top_k: [
            _community_doc(0, 0, ["qwen", "dashscope"],
                           summary="QWEN_ECOSYSTEM_SUMMARY"),
        ],
    )

    hits = retrieve_local("about qwen", kb_name="kb", store=store)

    community_hits = [h for h in hits if h.kind == "community"]
    assert len(community_hits) == 1
    assert community_hits[0].content == "QWEN_ECOSYSTEM_SUMMARY"


def test_local_collects_neighbor_entities(tmp_path, monkeypatch):
    """Stream 3: NetworkX 1-hop neighbors of the seed."""
    store = _populated_store(tmp_path)
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_entities_by_vector",
        lambda kb, q, top_k: [_seed_entity_doc("qwen")],
    )
    monkeypatch.setattr("ragkit.core.graph.retriever.fetch_chunks_by_ids", lambda kb, ids: [])
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_entity_names",
        lambda kb, names, top_k: [],
    )

    hits = retrieve_local("about qwen", kb_name="kb", store=store)

    entity_hits = [h for h in hits if h.kind == "entity"]
    names = {h.extra["name"] for h in entity_hits}
    # qwen is the seed; dashscope is 1-hop neighbor → must appear.
    assert "dashscope" in names
    # qwen itself should NOT appear in the neighbor stream.
    assert "qwen" not in names


def test_local_collects_relations(tmp_path, monkeypatch):
    """Stream 4: relations where at least one endpoint is the seed."""
    store = _populated_store(tmp_path)
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_entities_by_vector",
        lambda kb, q, top_k: [_seed_entity_doc("qwen")],
    )
    monkeypatch.setattr("ragkit.core.graph.retriever.fetch_chunks_by_ids", lambda kb, ids: [])
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_entity_names",
        lambda kb, names, top_k: [],
    )

    hits = retrieve_local("about qwen", kb_name="kb", store=store)

    rel_hits = [h for h in hits if h.kind == "relation"]
    rel_pairs = {(h.extra["source"], h.extra["target"]) for h in rel_hits}
    assert ("qwen", "dashscope") in rel_pairs


def test_local_groups_streams_in_order(tmp_path, monkeypatch):
    """Output order: chunks → communities → entities → relations."""
    store = _populated_store(tmp_path)
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_entities_by_vector",
        lambda kb, q, top_k: [_seed_entity_doc("qwen", source_chunks=["c1"])],
    )
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.fetch_chunks_by_ids",
        lambda kb, ids: [{"_id": "c1", "doc_id": "d", "docnm_kwd": "r.pdf",
                          "content_with_weight": "T"}],
    )
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_entity_names",
        lambda kb, names, top_k: [_community_doc(0, 0, ["qwen", "dashscope"], "S")],
    )

    hits = retrieve_local("about qwen", kb_name="kb", store=store)

    kinds_in_order = [h.kind for h in hits]
    # We expect: chunk, community, entity*, relation*
    assert kinds_in_order[0] == "chunk"
    # community comes after chunks
    assert "community" in kinds_in_order
    assert kinds_in_order.index("community") > kinds_in_order.index("chunk")
    # entities come after communities
    if "entity" in kinds_in_order:
        assert kinds_in_order.index("entity") > kinds_in_order.index("community")


# ==========================================================================
# retrieve_global — Map-Reduce
# ==========================================================================


def test_global_returns_empty_when_no_community_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_vector",
        lambda kb, q, level, top_k: [],
    )
    assert retrieve_global("anything", kb_name="kb") == []


def test_global_runs_map_reduce_and_returns_points(tmp_path, monkeypatch):
    """Happy path: community vector search → map-reduce → rated points."""
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_vector",
        lambda kb, q, level, top_k: [
            _community_doc(0, 0, ["a", "b"], summary="S0"),
            _community_doc(1, 0, ["c", "d"], summary="S1"),
        ],
    )

    # Mock the map-reduce pipeline to return 2 rated points.
    def fake_run(question, community_reports, **kw):
        return [
            RatedPoint(point="POINT_A", rating=9),
            RatedPoint(point="POINT_B", rating=7),
        ]

    monkeypatch.setattr("ragkit.core.graph.retriever.run_global_search", fake_run)

    hits = retrieve_global("themes?", kb_name="kb")

    assert len(hits) == 2
    assert all(h.kind == "point" for h in hits)
    # Highest-rated point first (already sorted by map-reduce)
    assert hits[0].content == "POINT_A"
    assert hits[0].extra["rating"] == 9


def test_global_passes_level_filter_through(tmp_path, monkeypatch):
    """--level N must propagate to search_communities_by_vector."""
    captured = {}

    def fake_search(kb, question, *, level, top_k):
        captured["level"] = level
        return []

    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_vector", fake_search
    )

    retrieve_global("q", kb_name="kb", level=2)
    assert captured["level"] == 2

    captured.clear()
    retrieve_global("q", kb_name="kb")  # default
    assert captured["level"] is None


def test_global_returns_empty_when_all_points_below_threshold(tmp_path, monkeypatch):
    """Map-Reduce returned no surviving points → empty result, not crash."""
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.search_communities_by_vector",
        lambda kb, q, level, top_k: [_community_doc(0, 0, ["a"], "S")],
    )
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.run_global_search",
        lambda *a, **kw: [],
    )
    assert retrieve_global("q", kb_name="kb") == []


# ==========================================================================
# Adapter: graph_hits_to_chunks
# ==========================================================================


def test_graph_hits_to_chunks_preserves_rank_and_kind(tmp_path):
    hits = [
        GraphHit(rank=1, kind="chunk", title="r.pdf", content="x", extra={"document_id": "d"}),
        GraphHit(rank=2, kind="entity", title="qwen [model]", content="y", extra={}),
        GraphHit(rank=3, kind="point", title="Point (rating 9/10)", content="z", extra={"rating": 9}),
    ]
    chunks = graph_hits_to_chunks(hits)
    assert [c.rank for c in chunks] == [1, 2, 3]
    # First chunk's document_id was preserved through extra.
    assert chunks[0].document_id == "d"


# ==========================================================================
# Regression: hybrid mode is removed
# ==========================================================================


def test_retrieve_hybrid_no_longer_exists():
    """hybrid was removed in task #25 — importing it should fail."""
    with pytest.raises(ImportError):
        from ragkit.core.graph.retriever import retrieve_hybrid  # noqa: F401
