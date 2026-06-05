"""Vector/keyword search helpers over the {kb}_graph index and source
chunks. Used by the new local/global retrievers (task #25).

These thin wrappers issue raw ES queries instead of going through the
heavier Dealer pipeline (which is BM25+dense fusion designed for chunks).
For graph artifacts we know exactly what shape the docs are, so simple
kNN + term filters are more direct.
"""

from __future__ import annotations

from typing import Any

from ragkit.core.embedder import embed_one
from ragkit.core.rag.utils.es_conn import ESConnection
from ragkit.logger import logger


def _graph_index(kb_name: str) -> str:
    return f"{kb_name}_graph"


def _vector_field_name(dim: int) -> str:
    """ES mapping uses dimension-tagged field names (q_1024_vec)."""
    return f"q_{dim}_vec"


def search_entities_by_vector(
    kb_name: str,
    query_text: str,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """kNN search over entity docs in {kb}_graph.

    Returns the raw _source dicts (caller decides which fields to use).
    """
    es = ESConnection().es
    index = _graph_index(kb_name)
    if not es.indices.exists(index=index):
        return []

    query_vector = embed_one(query_text)
    # ISS-016: surface embedding failures instead of silently returning [].
    if not query_vector:
        logger.warning(
            "search_entities_by_vector: embed_one returned an empty vector "
            "(API error or rate-limited). Returning no results."
        )
        return []
    field = _vector_field_name(len(query_vector))

    try:
        resp = es.search(
            index=index,
            knn={
                "field": field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
                "filter": {"term": {"type_kwd": "entity"}},
            },
            size=top_k,
        )
    except Exception as e:
        logger.warning(f"search_entities_by_vector failed on {index}: {e}")
        return []

    return [h.get("_source", {}) for h in resp.get("hits", {}).get("hits", [])]


def search_communities_by_vector(
    kb_name: str,
    query_text: str,
    *,
    level: int | None = None,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """kNN search over community docs in {kb}_graph.

    If ``level`` is provided, the search is restricted to that hierarchy
    level via a term filter. Otherwise communities from all levels compete.
    """
    es = ESConnection().es
    index = _graph_index(kb_name)
    if not es.indices.exists(index=index):
        return []

    query_vector = embed_one(query_text)
    # ISS-016: surface embedding failures instead of silently returning [].
    if not query_vector:
        logger.warning(
            "search_communities_by_vector: embed_one returned an empty vector. "
            "Returning no results."
        )
        return []
    field = _vector_field_name(len(query_vector))

    # Filter combines type AND optional level.
    must_filter: list[dict] = [{"term": {"type_kwd": "community"}}]
    if level is not None:
        must_filter.append({"term": {"community_level_int": level}})

    try:
        resp = es.search(
            index=index,
            knn={
                "field": field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
                "filter": {"bool": {"must": must_filter}},
            },
            size=top_k,
        )
    except Exception as e:
        logger.warning(f"search_communities_by_vector failed on {index}: {e}")
        return []

    return [h.get("_source", {}) for h in resp.get("hits", {}).get("hits", [])]


def search_communities_by_entity_names(
    kb_name: str,
    entity_names: list[str],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find community docs whose ``community_entity_names_kwd`` overlaps
    the given list. Used by local search to surface "which communities
    contain my seed entities".

    Returns up to ``top_k`` community docs, ordered by how many of the
    given names they contain (via ES `terms` aggregation behavior — we
    just match-any here and let community_rank_flt break ties).
    """
    es = ESConnection().es
    index = _graph_index(kb_name)
    if not entity_names or not es.indices.exists(index=index):
        return []

    try:
        resp = es.search(
            index=index,
            query={
                "bool": {
                    "must": [{"term": {"type_kwd": "community"}}],
                    "should": [
                        {"terms": {"community_entity_names_kwd": entity_names}}
                    ],
                    "minimum_should_match": 1,
                }
            },
            sort=[
                {"_score": {"order": "desc"}},
                {"community_rank_flt": {"order": "desc"}},
            ],
            size=top_k,
        )
    except Exception as e:
        logger.warning(f"search_communities_by_entity_names failed on {index}: {e}")
        return []

    return [h.get("_source", {}) for h in resp.get("hits", {}).get("hits", [])]


def fetch_chunks_by_ids(kb_name: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
    """mget chunks from the main {kb} index by their ES document IDs.

    Returns the _source dicts of chunks that exist; missing IDs are
    silently skipped (graph entity.source_chunks may reference chunks
    that have since been deleted).
    """
    es = ESConnection().es
    if not chunk_ids or not es.indices.exists(index=kb_name):
        return []

    try:
        # ISS-037: preserve order while deduplicating. set() would randomize.
        # dict.fromkeys keeps first-occurrence order (Py 3.7+ insertion order).
        unique_ids = list(dict.fromkeys(chunk_ids))
        resp = es.mget(index=kb_name, ids=unique_ids)
    except Exception as e:
        logger.warning(f"fetch_chunks_by_ids failed on {kb_name}: {e}")
        return []

    out: list[dict[str, Any]] = []
    for doc in resp.get("docs", []):
        if doc.get("found") and "_source" in doc:
            src = doc["_source"]
            src["_id"] = doc.get("_id")
            out.append(src)
    return out
