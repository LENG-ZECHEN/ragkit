"""Knowledge base (Elasticsearch index) management."""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.core._ragflow.rag.utils.es_conn import ESConnection


@dataclass(frozen=True)
class KbInfo:
    name: str
    document_count: int
    chunk_count: int


def list_kbs() -> list[str]:
    """Return all knowledge base names."""
    return ESConnection().list_indices()


def kb_info(name: str) -> KbInfo:
    """Stats for one knowledge base. Returns zeros if it does not exist."""
    es = ESConnection().es
    if not es.indices.exists(index=name):
        return KbInfo(name=name, document_count=0, chunk_count=0)

    # Count unique documents and total chunks.
    agg = es.search(
        index=name,
        body={
            "size": 0,
            "aggs": {"docs": {"cardinality": {"field": "doc_id"}}},
        },
    )
    return KbInfo(
        name=name,
        document_count=int(agg["aggregations"]["docs"]["value"]),
        chunk_count=int(agg["hits"]["total"]["value"]),
    )


def kb_documents(name: str) -> list[dict]:
    """Return one entry per unique document name in the KB."""
    es = ESConnection().es
    if not es.indices.exists(index=name):
        return []

    resp = es.search(
        index=name,
        body={
            "size": 0,
            "aggs": {
                "docs": {
                    "terms": {"field": "docnm_kwd", "size": 1000},
                }
            },
        },
    )
    return [
        {"document_name": b["key"], "chunks": b["doc_count"]}
        for b in resp["aggregations"]["docs"]["buckets"]
    ]


def delete_kb(name: str) -> bool:
    """Drop the knowledge base entirely. Returns True if it existed."""
    return ESConnection().delete_index(name)
