"""Graph data types — merging semantics matter because the same entity
appears across many chunks and we accumulate descriptions/source_chunks."""

from __future__ import annotations

from ragkit.core.graph.types import Entity, Relation


def test_entity_merge_unions_types_when_different():
    """Same entity flagged as different types (the LLM is inconsistent) →
    we keep both rather than picking, so users see the disagreement."""
    a = Entity(name="qwen", type="model", description="LLM made by Alibaba")
    b = Entity(name="qwen", type="product", description="hosted on DashScope")

    a.merge(b)

    assert "model" in a.type and "product" in a.type
    assert "DashScope" in a.description
    assert "Alibaba" in a.description


def test_entity_merge_dedupes_source_chunks():
    """Indexer may re-process a chunk; same source_chunk shouldn't be added twice."""
    a = Entity(name="x", type="t", source_chunks=["c1", "c2"])
    b = Entity(name="x", type="t", source_chunks=["c2", "c3"])

    a.merge(b)

    assert a.source_chunks == ["c1", "c2", "c3"]


def test_entity_merge_skips_duplicate_description():
    """The same description text shouldn't pile up across mentions."""
    a = Entity(name="x", type="t", description="A core concept.")
    b = Entity(name="x", type="t", description="A core concept.")

    a.merge(b)

    # No duplicate concatenation
    assert a.description == "A core concept."


def test_relation_merge_accumulates_weight():
    """Edge weight = co-mention frequency — must be additive across observations."""
    r = Relation(source="a", target="b", weight=1.0, source_chunks=["c1"])
    r.merge(Relation(source="a", target="b", weight=1.0, source_chunks=["c2"]))
    r.merge(Relation(source="a", target="b", weight=1.0, source_chunks=["c3"]))

    assert r.weight == 3.0
    assert set(r.source_chunks) == {"c1", "c2", "c3"}
