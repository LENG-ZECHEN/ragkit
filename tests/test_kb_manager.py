"""Knowledge-base manager — list / info / delete contract."""

from __future__ import annotations

import pytest

from ragkit.core.kb_manager import KbInfo, delete_kb, kb_documents, kb_info, list_kbs


def test_list_kbs_returns_index_names(fake_es):
    fake_es.list_indices.return_value = ["finance", "papers", "personal"]
    assert list_kbs() == ["finance", "papers", "personal"]


def test_kb_info_zero_when_missing(fake_es):
    """A KB the user typoed shouldn't crash — return zeros so the CLI
    can tell them 'no such knowledge base' instead of a traceback."""
    fake_es.es.indices.exists.return_value = False
    info = kb_info("nope")
    assert info == KbInfo(name="nope", document_count=0, chunk_count=0)


def test_kb_info_aggregates_from_es(fake_es):
    """Verify we use the right aggregation shape (cardinality on doc_id,
    total hits for chunks). If this breaks, the user sees wrong stats."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {
        "hits": {"total": {"value": 87}},
        "aggregations": {"docs": {"value": 3}},
    }

    info = kb_info("kb1")

    assert info.document_count == 3
    assert info.chunk_count == 87

    # Confirm we requested an aggregation, not a full scan.
    body = fake_es.es.search.call_args.kwargs["body"]
    assert body["size"] == 0
    assert "docs" in body["aggs"]


def test_kb_documents_returns_per_document_chunk_counts(fake_es):
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {
        "aggregations": {
            "docs": {
                "buckets": [
                    {"key": "report.pdf", "doc_count": 42},
                    {"key": "memo.docx", "doc_count": 8},
                ]
            }
        }
    }
    docs = kb_documents("kb")

    assert docs == [
        {"document_name": "report.pdf", "chunks": 42},
        {"document_name": "memo.docx", "chunks": 8},
    ]


def test_delete_kb_returns_existence_flag(fake_es):
    fake_es.delete_index.return_value = True
    assert delete_kb("kb") is True

    fake_es.delete_index.return_value = False
    assert delete_kb("ghost") is False
