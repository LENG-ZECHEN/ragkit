"""Graph RAG retrieval strategies — Microsoft GraphRAG style.

Two modes (hybrid was removed — local already does multi-source aggregation):

  local   — Entity-centric multi-source retrieval. Steps:
              1. Vector-search the {kb}_graph index for seed entities
              2. For each seed, gather 4 candidate streams:
                   - text units (raw chunks via source_chunks → {kb} index)
                   - community reports containing the entity
                   - 1-hop neighbor entities (from NetworkX)
                   - relations involving the entity (from NetworkX)
              3. Rank + filter each stream independently
              4. Return a unified GraphHit list grouped by stream

  global  — Map-Reduce over community reports. Steps:
              1. Vector-search {kb}_graph for top-K community reports
                 (filter by --level if specified, else cross-level)
              2. Shuffle + batch under token budget
              3. MAP: each batch → LLM → list of (point, rating)
              4. REDUCE: filter low-rated, return top-N
              5. Return GraphHit list, one per surviving rated point

To swap the LLM provider, see global_search.py:_client(). To change ranking
heuristics, edit the _rank_* functions in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragkit.core.graph.global_search import (
    RatedPoint,
    run_global_search,
)
from ragkit.core.graph.searcher import (
    fetch_chunks_by_ids,
    search_communities_by_entity_names,
    search_communities_by_vector,
    search_entities_by_vector,
)
from ragkit.core.graph.store import GraphStore, open_store
from ragkit.core.retriever import RetrievedChunk
from ragkit.logger import logger


# Per-stream caps for local search. Total context = sum of all streams.
LOCAL_TOP_K_SEEDS = 10
LOCAL_TOP_K_TEXT_UNITS = 5
LOCAL_TOP_K_COMMUNITIES = 3
LOCAL_TOP_K_ENTITIES = 8
LOCAL_TOP_K_RELATIONS = 8

# Global search defaults.
GLOBAL_TOP_K_REPORTS = 30


def _validate_args(question: str, kb_name: str) -> None:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not kb_name or not kb_name.strip():
        raise ValueError("kb_name must be a non-empty string")


@dataclass(frozen=True)
class GraphHit:
    """One retrieved evidence item.

    The ``kind`` field tells consumers what's inside:
      "chunk"     — original document text from {kb}
      "entity"    — entity description + type
      "relation"  — relation description between two entities
      "community" — full community report
      "point"     — a single rated point from global search reduce
    """

    rank: int
    kind: str
    title: str
    content: str
    extra: dict

    def as_chunk(self) -> RetrievedChunk:
        """Adapter so the existing generator can consume GraphHit lists."""
        return RetrievedChunk(
            rank=self.rank,
            document_id=self.extra.get("document_id", self.kind),
            document_name=self.title,
            content=self.content,
            similarity=float(self.extra.get("similarity", 0.0)),
            vector_similarity=0.0,
            term_similarity=0.0,
        )


# ==========================================================================
# LOCAL SEARCH — multi-source candidate collection
# ==========================================================================


def _collect_text_units(
    seed_entity_docs: list[dict[str, Any]],
    kb_name: str,
    *,
    top_k: int,
) -> list[GraphHit]:
    """Stream 1: chunks of original text mentioning the seed entities.

    Source: entity.source_chunks → {kb} index. Sorted by how many seed
    entities each chunk is associated with (more hits = more relevant).
    """
    # Build chunk_id → seed-entity hit count.
    chunk_hits: dict[str, int] = {}
    for entity in seed_entity_docs:
        for chunk_id in entity.get("source_chunks_kwd", []) or []:
            chunk_hits[chunk_id] = chunk_hits.get(chunk_id, 0) + 1

    if not chunk_hits:
        return []

    # Sort by hit count and fetch the top candidates from ES.
    ranked_ids = sorted(chunk_hits.keys(), key=lambda c: -chunk_hits[c])
    fetched = fetch_chunks_by_ids(kb_name, ranked_ids[: top_k * 2])

    # Preserve our ranking order in the output.
    by_id = {c.get("_id"): c for c in fetched}
    hits: list[GraphHit] = []
    for chunk_id in ranked_ids:
        chunk = by_id.get(chunk_id)
        if not chunk:
            continue
        hits.append(GraphHit(
            rank=len(hits) + 1,
            kind="chunk",
            title=chunk.get("docnm_kwd", "").split("/")[-1] or "(unknown)",
            content=chunk.get("content_with_weight", ""),
            extra={
                "document_id": chunk.get("doc_id", ""),
                "source_hits": chunk_hits[chunk_id],
            },
        ))
        if len(hits) >= top_k:
            break
    return hits


def _collect_communities(
    seed_entity_docs: list[dict[str, Any]],
    kb_name: str,
    *,
    top_k: int,
) -> list[GraphHit]:
    """Stream 2: community reports containing the seed entities.

    Source: {kb}_graph type=community with entity_names overlapping seeds.
    Sorted by ES score (more entity matches) and community_rank_flt.
    """
    seed_names = [e.get("entity_name_kwd", "") for e in seed_entity_docs]
    seed_names = [n for n in seed_names if n]
    if not seed_names:
        return []

    community_docs = search_communities_by_entity_names(
        kb_name, seed_names, top_k=top_k
    )
    return [
        GraphHit(
            rank=i,
            kind="community",
            title=c.get("content_with_weight", "").split("\n", 1)[0] or f"Community {c.get('community_id_int', '?')}",
            content=c.get("content_with_weight", ""),
            extra={
                "level": c.get("community_level_int"),
                "community_id": c.get("community_id_int"),
                "rank": c.get("community_rank_flt", 0.0),
            },
        )
        for i, c in enumerate(community_docs, start=1)
    ]


def _collect_neighbor_entities(
    seed_entity_docs: list[dict[str, Any]],
    store: GraphStore,
    *,
    top_k: int,
) -> list[GraphHit]:
    """Stream 3: 1-hop neighbor entities (from local NetworkX graph)."""
    seed_names = {e.get("entity_name_kwd", "").lower() for e in seed_entity_docs}
    seed_names.discard("")

    collected: dict[str, Any] = {}
    for name in seed_names:
        for nb in store.neighbors(name, depth=1):
            # Skip seeds themselves; they're already covered.
            if nb.name in seed_names or nb.name in collected:
                continue
            collected[nb.name] = nb

    # Rank neighbors by mention frequency (proxy for importance).
    ranked = sorted(
        collected.values(),
        key=lambda e: -len(e.source_chunks),
    )
    return [
        GraphHit(
            rank=i,
            kind="entity",
            title=f"{e.name} [{e.type}]",
            content=e.description or "",
            extra={
                "name": e.name,
                "source_chunks": e.source_chunks,
            },
        )
        for i, e in enumerate(ranked[:top_k], start=1)
    ]


def _collect_relations(
    seed_entity_docs: list[dict[str, Any]],
    store: GraphStore,
    *,
    top_k: int,
) -> list[GraphHit]:
    """Stream 4: edges where at least one endpoint is a seed entity.

    Sorted by edge weight (co-mention count, an importance proxy).
    """
    seed_names = {e.get("entity_name_kwd", "").lower() for e in seed_entity_docs}
    seed_names.discard("")

    candidates = []
    for r in store.all_relations():
        if r.source in seed_names or r.target in seed_names:
            candidates.append(r)

    candidates.sort(key=lambda r: -r.weight)
    return [
        GraphHit(
            rank=i,
            kind="relation",
            title=f"{r.source} ↔ {r.target}",
            content=r.description or "",
            extra={
                "source": r.source,
                "target": r.target,
                "weight": r.weight,
            },
        )
        for i, r in enumerate(candidates[:top_k], start=1)
    ]


def retrieve_local(
    question: str,
    kb_name: str,
    *,
    top_k: int = 5,  # legacy param — caller-visible cap on RESULT stream sizes
    store: GraphStore | None = None,
) -> list[GraphHit]:
    """Microsoft-style local retrieval (4 candidate streams + ranking).

    ``top_k`` is loosely applied: per-stream defaults are used (LOCAL_TOP_K_*)
    but if the caller passes a smaller value we honor it on each stream.

    Returns a single flat GraphHit list, grouped by stream (chunks first,
    then communities, neighbors, relations), with re-numbered global ranks.
    """
    _validate_args(question, kb_name)
    store = store or open_store(kb_name)

    # ---- 1. Find seed entities via vector search ----
    seed_docs = search_entities_by_vector(
        kb_name, question, top_k=LOCAL_TOP_K_SEEDS
    )
    if not seed_docs:
        logger.info(f"Local search: no seed entities found for query in {kb_name}_graph")
        return []

    # ---- 2. Multi-source candidate collection ----
    text_units = _collect_text_units(seed_docs, kb_name, top_k=min(top_k, LOCAL_TOP_K_TEXT_UNITS))
    communities = _collect_communities(seed_docs, kb_name, top_k=min(top_k, LOCAL_TOP_K_COMMUNITIES))
    neighbors = _collect_neighbor_entities(seed_docs, store, top_k=LOCAL_TOP_K_ENTITIES)
    relations = _collect_relations(seed_docs, store, top_k=LOCAL_TOP_K_RELATIONS)

    # ---- 3. Concatenate streams with renumbered ranks ----
    all_hits = text_units + communities + neighbors + relations
    return [
        GraphHit(
            rank=i,
            kind=h.kind,
            title=h.title,
            content=h.content,
            extra=h.extra,
        )
        for i, h in enumerate(all_hits, start=1)
    ]


# ==========================================================================
# GLOBAL SEARCH — Map-Reduce over community reports
# ==========================================================================


def retrieve_global(
    question: str,
    kb_name: str,
    *,
    level: int | None = None,
    top_k: int = 20,
    store: GraphStore | None = None,
) -> list[GraphHit]:
    """Microsoft-style global retrieval.

    Args:
        question: user query
        kb_name: knowledge base
        level: optional community level filter (None = cross-level vector search)
        top_k: max final points to return
        store: unused here (kept for signature consistency)

    Implementation:
      1. Vector-search up to GLOBAL_TOP_K_REPORTS communities in {kb}_graph
      2. run_global_search performs Map-Reduce → list[RatedPoint]
      3. Wrap surviving points as GraphHit
    """
    _validate_args(question, kb_name)

    # 1. Candidate community reports.
    community_docs = search_communities_by_vector(
        kb_name, question, level=level, top_k=GLOBAL_TOP_K_REPORTS
    )
    if not community_docs:
        logger.info(f"Global search: no community reports for query in {kb_name}_graph")
        return []

    # 2. Map-Reduce.
    rated_points = run_global_search(question, community_docs, final_top_k=top_k)
    if not rated_points:
        return []

    # 3. Wrap as GraphHit (kind="point" so consumers know this is a
    #    map-reduce output, not a raw document).
    return [
        GraphHit(
            rank=i,
            kind="point",
            title=f"Point (rating {p.rating}/10)",
            content=p.point,
            extra={"rating": p.rating, "source": p.source},
        )
        for i, p in enumerate(rated_points, start=1)
    ]


# ==========================================================================
# Adapter so the existing generator can consume GraphHit lists.
# ==========================================================================


def graph_hits_to_chunks(hits: list[GraphHit]) -> list[RetrievedChunk]:
    """Convert GraphHits → RetrievedChunks (one shape the generator knows)."""
    return [h.as_chunk() for h in hits]
