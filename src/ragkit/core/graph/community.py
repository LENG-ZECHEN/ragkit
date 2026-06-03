"""Community detection — cluster the knowledge graph into related groups.

Default algorithm: Louvain (via python-louvain).
To swap: change `detect_communities` body. The graph store API stays the same.
"""

from __future__ import annotations

from collections import defaultdict

import community as community_louvain  # python-louvain package

from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Community
from ragkit.logger import logger

# Communities with fewer than this many members are merged into one bucket.
# Otherwise the long tail of singletons creates noise for the global query.
MIN_COMMUNITY_SIZE = 2


def detect_communities(store: NetworkXGraphStore, *, seed: int = 42) -> list[Community]:
    """Run Louvain on the graph and return Communities.

    Args:
        store: graph backend
        seed: random_state for reproducibility

    Returns:
        Communities sorted by size (largest first). Empty list if no edges.
    """
    g = store.g
    if g.number_of_edges() == 0:
        # No edges → no Louvain partition. But we still want isolated entities
        # discoverable via global retrieval. Bundle them into one misc community.
        if g.number_of_nodes() == 0:
            logger.info("Graph is empty — no communities")
            return []
        logger.info("Graph has no edges — bundling isolated nodes into misc community")
        return [Community(
            id=0,
            entity_names=sorted(g.nodes()),
            extra={"is_misc_bucket": True},
        )]

    partition = community_louvain.best_partition(g, random_state=seed, weight="weight")

    # Group nodes by their community id.
    by_cid: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        by_cid[cid].append(node)

    # Merge undersized communities into one bucket so they're not noise.
    communities: list[Community] = []
    misc_members: list[str] = []
    next_id = 0
    for members in by_cid.values():
        if len(members) >= MIN_COMMUNITY_SIZE:
            communities.append(Community(id=next_id, entity_names=sorted(members)))
            next_id += 1
        else:
            misc_members.extend(members)

    if misc_members:
        communities.append(Community(
            id=next_id,
            entity_names=sorted(misc_members),
            extra={"is_misc_bucket": True},
        ))

    communities.sort(key=lambda c: len(c.entity_names), reverse=True)
    # Reassign IDs after sort so id 0 is always the biggest.
    for i, c in enumerate(communities):
        c.id = i

    logger.info(f"Detected {len(communities)} communities from {g.number_of_nodes()} entities")
    return communities
