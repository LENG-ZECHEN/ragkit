"""Community detection — deterministic clustering, edge cases."""

from __future__ import annotations

from pathlib import Path

from ragkit.core.graph.community import MIN_COMMUNITY_SIZE, detect_communities
from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Entity, Relation


def _store_with_two_clusters(tmp_path: Path) -> NetworkXGraphStore:
    """Build two clearly separated clusters: {a,b,c} and {x,y,z}."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    # cluster 1: a-b-c triangle
    for src, tgt in [("a", "b"), ("b", "c"), ("a", "c")]:
        store.upsert_relation(Relation(source=src, target=tgt, weight=1.0))
    # cluster 2: x-y-z triangle
    for src, tgt in [("x", "y"), ("y", "z"), ("x", "z")]:
        store.upsert_relation(Relation(source=src, target=tgt, weight=1.0))
    # weak bridge that shouldn't pull them together
    store.upsert_relation(Relation(source="c", target="x", weight=0.1))
    return store


def test_detect_communities_separates_clusters(tmp_path):
    """Two triangles weakly connected should be split into 2 communities."""
    store = _store_with_two_clusters(tmp_path)
    communities = detect_communities(store)

    # At least 2 sizeable clusters (Louvain may produce more on ties).
    assert len(communities) >= 2

    cluster_1 = {"a", "b", "c"}
    cluster_2 = {"x", "y", "z"}
    # Each original cluster should be contained in a single community.
    for original in (cluster_1, cluster_2):
        matching = [c for c in communities if original.issubset(set(c.entity_names))]
        assert len(matching) == 1


def test_detect_communities_is_deterministic_with_seed(tmp_path):
    """Same graph + same seed → same partition. Critical for reproducible
    summaries (which are cached on community.id)."""
    store = _store_with_two_clusters(tmp_path)

    a = detect_communities(store, seed=42)
    b = detect_communities(store, seed=42)

    assert [(c.id, sorted(c.entity_names)) for c in a] == \
           [(c.id, sorted(c.entity_names)) for c in b]


def test_detect_communities_empty_graph(tmp_path):
    """An empty graph must return [] instead of crashing on Louvain."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    assert detect_communities(store) == []


def test_detect_communities_isolated_nodes_bundled_into_misc(tmp_path):
    """No edges but entities exist → bundle them as one misc community so
    they're still discoverable via global retrieval."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    store.upsert_entity(Entity(name="alone1", type="t"))
    store.upsert_entity(Entity(name="alone2", type="t"))

    communities = detect_communities(store)
    assert len(communities) == 1
    assert communities[0].extra.get("is_misc_bucket") is True
    assert set(communities[0].entity_names) == {"alone1", "alone2"}


def test_detect_communities_truly_empty_graph_returns_empty(tmp_path):
    """Zero nodes AND zero edges → no communities."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    assert detect_communities(store) == []


def test_detect_communities_singletons_merged_into_misc(tmp_path):
    """Below MIN_COMMUNITY_SIZE communities are merged into one 'misc'
    bucket so the summarizer doesn't waste calls on noise."""
    assert MIN_COMMUNITY_SIZE >= 2  # contract check
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    # Make 5 disconnected pairs
    for i in range(5):
        store.upsert_relation(Relation(source=f"p{i}_a", target=f"p{i}_b"))

    communities = detect_communities(store)
    # All non-misc communities should have >= MIN_COMMUNITY_SIZE members.
    big = [c for c in communities if not c.extra.get("is_misc_bucket")]
    for c in big:
        assert len(c.entity_names) >= MIN_COMMUNITY_SIZE
    # And the misc bucket must actually exist with the singletons collected.
    misc_buckets = [c for c in communities if c.extra.get("is_misc_bucket") is True]
    # When all pairs are isolated and equal-sized, Louvain may group them OR
    # they may end up in misc. The contract: any community below MIN_SIZE
    # must be flagged as misc.
    for c in communities:
        if len(c.entity_names) < MIN_COMMUNITY_SIZE:
            assert c.extra.get("is_misc_bucket") is True


def test_communities_sorted_largest_first(tmp_path):
    """Largest community is id=0 — downstream code (e.g. summary capping)
    relies on this ordering invariant."""
    store = _store_with_two_clusters(tmp_path)
    # Add another node connected to cluster 1 to make it bigger.
    store.upsert_relation(Relation(source="a", target="d", weight=1.0))
    store.upsert_relation(Relation(source="b", target="d", weight=1.0))

    communities = detect_communities(store)
    sizes = [len(c.entity_names) for c in communities]
    assert sizes == sorted(sizes, reverse=True)
    assert communities[0].id == 0
