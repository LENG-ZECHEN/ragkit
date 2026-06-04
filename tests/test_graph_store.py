"""Graph store — persistence roundtrip, merge semantics, neighborhood."""

from __future__ import annotations

import json
from pathlib import Path

from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Community, Entity, Relation


def _make_store(tmp_path: Path) -> NetworkXGraphStore:
    return NetworkXGraphStore(path=tmp_path / "graph.json")


def test_upsert_entity_normalizes_name_to_lowercase(tmp_path):
    """Entity lookup is case-insensitive — 'OpenAI' and 'openai' must be one node."""
    store = _make_store(tmp_path)
    store.upsert_entity(Entity(name="OpenAI", type="org", description="A"))
    store.upsert_entity(Entity(name="openai", type="org", description="B"))

    assert store.entity_count() == 1
    e = store.get_entity("OPENAI")
    assert e is not None
    assert "A" in e.description and "B" in e.description


def test_upsert_relation_auto_creates_endpoints(tmp_path):
    """Adding an edge before its nodes shouldn't crash — auto-create them.
    Common when the LLM emits a relation referencing a stripped entity."""
    store = _make_store(tmp_path)
    store.upsert_relation(Relation(source="x", target="y", description="r"))

    assert store.entity_count() == 2
    assert store.relation_count() == 1


def test_upsert_relation_rejects_self_loops(tmp_path):
    """Self-loops break community detection — must drop silently."""
    store = _make_store(tmp_path)
    store.upsert_relation(Relation(source="a", target="a", description="self"))

    assert store.relation_count() == 0


def test_neighbors_bfs_at_specified_depth(tmp_path):
    """depth=2 should reach grandchildren but stop there."""
    store = _make_store(tmp_path)
    # Chain: a — b — c — d
    store.upsert_relation(Relation(source="a", target="b"))
    store.upsert_relation(Relation(source="b", target="c"))
    store.upsert_relation(Relation(source="c", target="d"))

    depth_1 = {e.name for e in store.neighbors("a", depth=1)}
    depth_2 = {e.name for e in store.neighbors("a", depth=2)}
    depth_3 = {e.name for e in store.neighbors("a", depth=3)}

    assert depth_1 == {"b"}
    assert depth_2 == {"b", "c"}
    assert depth_3 == {"b", "c", "d"}
    # Self never appears
    assert "a" not in depth_3


def test_neighbors_unknown_node_returns_empty(tmp_path):
    """Don't crash on questions that mention entities the graph doesn't have."""
    store = _make_store(tmp_path)
    store.upsert_entity(Entity(name="alpha", type="t"))
    assert store.neighbors("does-not-exist") == []


def test_save_load_roundtrip(tmp_path):
    """Saved graph + reopened store must contain exactly what we wrote."""
    path = tmp_path / "g.json"
    s1 = NetworkXGraphStore(path)
    s1.upsert_entity(Entity(name="alpha", type="concept", description="An α."))
    s1.upsert_entity(Entity(name="beta", type="concept", description="A β."))
    s1.upsert_relation(Relation(source="alpha", target="beta", description="related", weight=2.5))
    s1.set_communities([Community(id=0, entity_names=["alpha", "beta"], summary="Greek letters.")])
    s1.save()

    # Re-open
    s2 = NetworkXGraphStore(path)
    assert s2.entity_count() == 2
    assert s2.relation_count() == 1
    alpha = s2.get_entity("alpha")
    assert alpha is not None and alpha.type == "concept"
    edges = list(s2.all_relations())
    assert edges[0].weight == 2.5
    communities = s2.all_communities()
    assert communities[0].summary == "Greek letters."


def test_save_writes_valid_json(tmp_path):
    """Output must be parseable JSON — guards against tuple/set leaking in."""
    path = tmp_path / "g.json"
    store = NetworkXGraphStore(path)
    store.upsert_entity(Entity(name="x", type="t"))
    store.save()
    payload = json.loads(path.read_text())
    assert "entities" in payload and "relations" in payload


def test_load_handles_corrupt_file_gracefully(tmp_path):
    """A corrupt save shouldn't crash app startup — and must NOT overwrite
    the corrupt file on next save (could destroy recoverable data)."""
    path = tmp_path / "g.json"
    path.write_text("{not valid json")

    store = NetworkXGraphStore(path)
    # Starts empty rather than raising
    assert store.entity_count() == 0
    # The corrupt file must be moved aside so a later save() doesn't blow it away.
    assert not path.exists()
    assert path.with_suffix(".json.corrupt").exists()


def test_double_roundtrip_preserves_data(tmp_path):
    """save → reload → mutate → save → reload again preserves everything.

    Critical because `_load_if_exists` uses `g.add_node` directly (not upsert),
    so we need to confirm a subsequent `upsert_entity` on a loaded node still
    merges instead of creating a duplicate.
    """
    path = tmp_path / "g.json"

    s1 = NetworkXGraphStore(path)
    s1.upsert_entity(Entity(name="x", type="t", description="first", source_chunks=["c1"]))
    s1.save()

    s2 = NetworkXGraphStore(path)
    # Upsert the same entity again with new info — must merge, not duplicate.
    s2.upsert_entity(Entity(name="X", type="t2", description="second", source_chunks=["c2"]))
    s2.save()

    s3 = NetworkXGraphStore(path)
    assert s3.entity_count() == 1
    e = s3.get_entity("x")
    assert e is not None
    assert "first" in e.description and "second" in e.description
    assert set(e.source_chunks) == {"c1", "c2"}


def test_clear_removes_data_and_file(tmp_path):
    path = tmp_path / "g.json"
    s = NetworkXGraphStore(path)
    s.upsert_entity(Entity(name="x", type="t"))
    s.save()
    assert path.exists()

    s.clear()
    assert s.entity_count() == 0
    assert not path.exists()


# ----- replace_*_description (direct override, bypassing merge) -----------


def test_replace_entity_description_overwrites_not_concatenates(tmp_path):
    """LLM consolidator depends on this: replace must REPLACE, not append."""
    store = _make_store(tmp_path)
    store.upsert_entity(Entity(name="x", type="t", description="A B C D E F"))
    store.replace_entity_description("x", "Z")
    e = store.get_entity("x")
    assert e is not None
    assert e.description == "Z"
    assert "A" not in e.description  # No concatenation happened


def test_replace_entity_description_unknown_entity_is_noop(tmp_path):
    """Calling replace on a missing entity must not crash (logs a warning)."""
    store = _make_store(tmp_path)
    store.replace_entity_description("ghost", "anything")
    assert store.entity_count() == 0


def test_replace_entity_description_is_case_insensitive(tmp_path):
    """The store normalizes names to lowercase; replace must match too."""
    store = _make_store(tmp_path)
    store.upsert_entity(Entity(name="OpenAI", type="org", description="orig"))
    store.replace_entity_description("OPENAI", "new")
    e = store.get_entity("openai")
    assert e is not None
    assert e.description == "new"


def test_replace_relation_description_overwrites(tmp_path):
    """Same semantics for relations."""
    store = _make_store(tmp_path)
    store.upsert_relation(Relation(source="a", target="b", description="orig"))
    store.replace_relation_description("a", "b", "new")
    edges = list(store.all_relations())
    assert len(edges) == 1
    assert edges[0].description == "new"


def test_replace_relation_description_unknown_edge_is_noop(tmp_path):
    """Calling replace on a missing edge must not crash."""
    store = _make_store(tmp_path)
    store.replace_relation_description("x", "y", "anything")
    assert store.relation_count() == 0


def test_replace_persists_through_save_load(tmp_path):
    """Replacement must survive serialization (no special field involved
    — but verify the description actually got persisted)."""
    path = tmp_path / "g.json"
    s1 = NetworkXGraphStore(path)
    s1.upsert_entity(Entity(name="x", type="t", description="orig"))
    s1.replace_entity_description("x", "rewritten")
    s1.save()

    s2 = NetworkXGraphStore(path)
    e = s2.get_entity("x")
    assert e is not None
    assert e.description == "rewritten"
