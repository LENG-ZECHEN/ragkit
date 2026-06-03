"""Retriever — validation, ES → RetrievedChunk mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragkit.core.retriever import RetrievedChunk, retrieve


def test_retrieve_rejects_empty_question():
    """An empty query is always a bug at the caller — fail loudly."""
    with pytest.raises(ValueError, match="non-empty"):
        retrieve("", kb_name="any")
    with pytest.raises(ValueError, match="non-empty"):
        retrieve("   ", kb_name="any")


def test_retrieve_maps_es_chunks_to_dataclass(monkeypatch):
    """The internal Dealer returns dict-shaped chunks. retrieve() must
    expose them as immutable RetrievedChunk dataclasses with stable fields."""
    fake_dealer = MagicMock()
    fake_dealer.retrieval.return_value = {
        "chunks": [
            {
                "doc_id": "doc-A",
                "docnm_kwd": "/some/path/report.pdf",
                "content_with_weight": "Alpha content.",
                "similarity": 0.91,
                "vector_similarity": 0.88,
                "term_similarity": 0.93,
            },
            {
                "doc_id": "doc-B",
                "docnm_kwd": "memo.docx",
                "content_with_weight": "Beta content.",
                "similarity": 0.75,
                "vector_similarity": 0.70,
                "term_similarity": 0.80,
            },
        ],
        "total": 2,
    }
    monkeypatch.setattr("ragkit.core.retriever._get_dealer", lambda: fake_dealer)

    results = retrieve("any question", kb_name="finance", top_k=5)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)

    # Ranks are 1-indexed.
    assert results[0].rank == 1
    assert results[1].rank == 2

    # Path is stripped to basename for display.
    assert results[0].document_name == "report.pdf"
    assert results[1].document_name == "memo.docx"

    # Similarity score preserved.
    assert results[0].similarity == pytest.approx(0.91)
    assert results[0].term_similarity == pytest.approx(0.93)


def test_retrieve_returns_empty_when_no_hits(monkeypatch):
    """No matches = empty list, not error (CLI prints a warning)."""
    fake_dealer = MagicMock()
    fake_dealer.retrieval.return_value = {"chunks": [], "total": 0}
    monkeypatch.setattr("ragkit.core.retriever._get_dealer", lambda: fake_dealer)

    results = retrieve("nothing in here", kb_name="empty_kb")
    assert results == []


def test_retrieve_chunk_is_immutable():
    """Frozen dataclass — downstream code must not mutate retrieved data."""
    c = RetrievedChunk(
        rank=1, document_id="x", document_name="x", content="x",
        similarity=0.0, vector_similarity=0.0, term_similarity=0.0,
    )
    with pytest.raises((AttributeError, Exception)):
        c.rank = 99  # type: ignore[misc]


def test_retrieve_passes_through_hybrid_weight(monkeypatch):
    """vector_similarity_weight is the main knob — verify it's forwarded."""
    fake_dealer = MagicMock()
    fake_dealer.retrieval.return_value = {"chunks": [], "total": 0}
    monkeypatch.setattr("ragkit.core.retriever._get_dealer", lambda: fake_dealer)

    retrieve("q", kb_name="kb", vector_similarity_weight=0.8)

    kwargs = fake_dealer.retrieval.call_args.kwargs
    assert kwargs["vector_similarity_weight"] == 0.8
    assert kwargs["tenant_ids"] == "kb"
