"""Indexer — the parse→chunk→embed→write pipeline that's hardest to get right."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragkit.core.indexer import _build_doc, index_file


# ----- _build_doc unit ----------------------------------------------------


def test_build_doc_assembles_all_required_es_fields():
    """ES mapping expects these exact keys — guard the contract."""
    chunk = {
        "content_with_weight": "Some content.",
        "content_ltks": "some content",
        "content_sm_ltks": "some content",
        "docnm_kwd": "file.pdf",
        "title_tks": "file",
    }
    doc = _build_doc(chunk, kb_name="kb1", file_name="file.pdf", embedding=[0.1, 0.2])

    # Identifiers
    assert "id" in doc and doc["id"]
    assert "doc_id" in doc and doc["doc_id"]
    assert doc["kb_id"] == "kb1"
    assert doc["docnm"] == "file.pdf"

    # Vector field name is dimension-tagged — `q_<dim>_vec` per the original schema.
    assert "q_2_vec" in doc
    assert doc["q_2_vec"] == [0.1, 0.2]

    # Content is preserved.
    assert doc["content_with_weight"] == "Some content."


def test_build_doc_chunk_id_is_deterministic():
    """Same content + same kb → same id, so re-indexing dedupes naturally."""
    chunk = {
        "content_with_weight": "Repeatable.",
        "content_ltks": "repeatable",
        "content_sm_ltks": "repeatable",
        "docnm_kwd": "f.pdf",
        "title_tks": "f",
    }
    d1 = _build_doc(chunk, "kb", "f.pdf", [0.0, 0.0])
    d2 = _build_doc(chunk, "kb", "f.pdf", [0.0, 0.0])
    assert d1["id"] == d2["id"]


def test_build_doc_chunk_id_differs_across_kbs():
    """Same content in two KBs must get distinct ids — prevents cross-kb leaks."""
    chunk = {
        "content_with_weight": "Same.",
        "content_ltks": "same",
        "content_sm_ltks": "same",
        "docnm_kwd": "f.pdf",
        "title_tks": "f",
    }
    d1 = _build_doc(chunk, "kb_a", "f.pdf", [0.0])
    d2 = _build_doc(chunk, "kb_b", "f.pdf", [0.0])
    assert d1["id"] != d2["id"]


# ----- end-to-end index_file with mocks ----------------------------------


def test_index_file_runs_full_pipeline(sample_txt, fake_openai, fake_es, monkeypatch):
    """Full pipeline: parse → chunk → embed → ensure_index → bulk insert."""
    result = index_file(sample_txt, kb_name="testkb")

    assert result["file"] == sample_txt.name
    assert result["kb"] == "testkb"
    assert result["chunks"] > 0

    # We MUST create the index before writing.
    fake_es.ensure_index.assert_called_once_with("testkb")
    fake_es.insert.assert_called_once()

    # Every inserted doc has a vector — that's the load-bearing invariant.
    insert_args = fake_es.insert.call_args
    docs = insert_args[0][0] if insert_args[0] else insert_args.kwargs["documents"]
    assert all(any(k.startswith("q_") and k.endswith("_vec") for k in d) for d in docs)


def test_index_file_aborts_on_es_errors(sample_txt, fake_openai, fake_es):
    """ES rejection must surface to the CLI as an exception — silent failures
    would leave users with a phantom-indexed file they can't query."""
    fake_es.insert.return_value = ["doc-x: bad mapping", "doc-y: shard failure"]

    with pytest.raises(RuntimeError, match="Failed to index"):
        index_file(sample_txt, kb_name="kb")


def test_index_file_progress_callback_fires(sample_txt, fake_openai, fake_es):
    """Progress UX is part of the contract — the CLI's progress bar needs it."""
    stages: list[str] = []

    def cb(stage: str, prog: float) -> None:
        stages.append(stage)
        assert 0.0 <= prog <= 1.0

    index_file(sample_txt, kb_name="kb", progress_cb=cb)

    # Order matters for the progress bar.
    assert stages[0] == "parsing"
    assert "embedding" in stages
    assert "indexing" in stages
    assert stages[-1] == "done"


def test_index_file_aborts_when_many_embeddings_fail(sample_txt, fake_openai, fake_es, monkeypatch):
    """If >10% of chunks fail to embed (sparse Nones), refuse to index a
    partial file silently — users wouldn't know which chunks went missing."""
    from ragkit.core import indexer

    def all_none(texts):
        # Every chunk failed → 100% failure → must abort
        return [None for _ in texts]

    monkeypatch.setattr(indexer, "embed_batch", all_none)

    with pytest.raises(RuntimeError, match="Embedding failed"):
        index_file(sample_txt, kb_name="kb")
