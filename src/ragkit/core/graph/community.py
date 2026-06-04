"""Community detection — cluster the knowledge graph into related groups.

Hierarchical version: uses Louvain's natural dendrogram (via
``community_louvain.generate_dendrogram``) to produce multiple levels of
granularity. Each Community carries a ``level`` field; level 0 is the
COARSEST (fewest, biggest groups), levels increase as the partition gets
finer.

To swap the algorithm (e.g. to Leiden via graspologic), replace
``detect_communities``'s body. The store/Community API stays the same.
"""

from __future__ import annotations

from collections import defaultdict

import community as community_louvain  # python-louvain package

from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Community
from ragkit.logger import logger

# Communities with fewer than this many members are merged into one
# "misc" bucket at each level so they don't pollute global retrieval.
MIN_COMMUNITY_SIZE = 2

# Safety cap on how many dendrogram levels we keep. Beyond 3 levels the
# granularity differences become marginal but the storage / summary cost
# keeps growing.
MAX_LEVELS = 3


def detect_communities(
    store: NetworkXGraphStore,
    *,
    seed: int = 42,
    max_levels: int = MAX_LEVELS,
) -> list[Community]:
    """Run hierarchical Louvain on the graph.

    Returns a flat ``list[Community]`` spanning all levels — distinguish
    them via the ``Community.level`` field.  Community IDs are globally
    unique counters across all levels (level 0 communities first, then
    level 1, ...).

    Edge cases:
      - Empty graph (no nodes, no edges)        → returns []
      - Graph with nodes but no edges           → returns one misc bucket at level 0
      - Dendrogram shallower than ``max_levels`` → we just stop at the real depth

    Args:
        store: graph backend
        seed: random_state for reproducibility (Louvain is randomized)
        max_levels: cap on hierarchy depth (default 3)
    """
    g = store.g

    if g.number_of_nodes() == 0:
        logger.info("Graph is empty — no communities")
        return []

    # No edges → no partition possible; bundle the isolated nodes so they
    # still appear in global retrieval.
    if g.number_of_edges() == 0:
        logger.info("Graph has no edges — bundling isolated nodes into misc community")
        return [
            Community(
                id=0,
                level=0,
                entity_names=sorted(g.nodes()),
                extra={"is_misc_bucket": True},
            )
        ]

    # generate_dendrogram returns a list where index 0 is the FINEST
    # partition and the last index is the COARSEST (smallest number of
    # groups). We invert that so our level 0 = coarsest, consistent with
    # the GraphRAG convention.
    dendro = community_louvain.generate_dendrogram(g, random_state=seed)
    if not dendro:
        # Fallback (shouldn't normally happen since we have edges) — use
        # best_partition for a single flat level.
        partition = community_louvain.best_partition(g, random_state=seed, weight="weight")
        return _partition_to_communities(partition, level=0, starting_id=0)

    n_levels = min(max_levels, len(dendro))

    # Walk from coarsest (top) to finest (bottom).
    # dendro index: 0=finest, len-1=coarsest. Reverse it.
    all_communities: list[Community] = []
    next_global_id = 0
    for our_level in range(n_levels):
        dendro_index = len(dendro) - 1 - our_level
        partition = community_louvain.partition_at_level(dendro, dendro_index)

        level_communities = _partition_to_communities(
            partition, level=our_level, starting_id=next_global_id
        )
        all_communities.extend(level_communities)
        next_global_id += len(level_communities)

    logger.info(
        f"Hierarchical detection: {len(all_communities)} communities across "
        f"{n_levels} level(s) from {g.number_of_nodes()} entities"
    )
    return all_communities


def _partition_to_communities(
    partition: dict[str, int], *, level: int, starting_id: int
) -> list[Community]:
    """Convert a flat ``{node: cid}`` partition into a sorted, ID-assigned
    list of Communities for a single level.

    Steps:
      1. Group nodes by partition cid
      2. Pool undersized groups into one misc bucket
      3. Sort largest-first
      4. Assign IDs starting from ``starting_id``
    """
    by_cid: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        by_cid[cid].append(node)

    communities: list[Community] = []
    misc_members: list[str] = []
    for members in by_cid.values():
        if len(members) >= MIN_COMMUNITY_SIZE:
            communities.append(
                Community(id=0, level=level, entity_names=sorted(members))
            )
        else:
            misc_members.extend(members)

    if misc_members:
        communities.append(
            Community(
                id=0,
                level=level,
                entity_names=sorted(misc_members),
                extra={"is_misc_bucket": True},
            )
        )

    # Largest first, then assign global IDs in that order.
    communities.sort(key=lambda c: len(c.entity_names), reverse=True)
    for i, c in enumerate(communities):
        c.id = starting_id + i
    return communities
