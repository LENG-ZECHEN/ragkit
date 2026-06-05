"""Indexer — the parse→chunk→embed→write pipeline that's hardest to get right."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragkit.core.indexer import (
    _build_doc,
    _count_existing_chunks_for_file,
    _delete_existing_chunks_for_file,
    index_file,
)


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


def test_index_file_build_graph_invokes_graph_builder(tmp_path, sample_txt, fake_openai, fake_es, monkeypatch):
    """`--build-graph` (build_graph=True) must call the graph builder with
    the indexed chunks AND surface entity/relation counts in the result."""
    from ragkit.core.graph.store import NetworkXGraphStore
    from ragkit.core.graph.types import Entity, Relation

    captured: dict = {}

    def fake_build_graph(chunks, kb_name, **kw):
        captured["n_chunks"] = len(list(chunks))
        captured["kb"] = kb_name
        store = NetworkXGraphStore(path=tmp_path / "fake_graph.json")
        store.upsert_entity(Entity(name="x", type="t"))
        store.upsert_entity(Entity(name="y", type="t"))
        store.upsert_relation(Relation(source="x", target="y"))
        return store

    monkeypatch.setattr("ragkit.core.graph.builder.build_graph", fake_build_graph)

    result = index_file(sample_txt, kb_name="kb", build_graph=True)

    assert captured["kb"] == "kb"
    assert captured["n_chunks"] >= 1
    assert result["graph_entities"] == 2
    assert result["graph_relations"] == 1


def test_index_file_no_graph_path_does_not_import_graph_builder(sample_txt, fake_openai, fake_es, monkeypatch):
    """When build_graph=False, the graph builder must NOT be invoked even
    once — this matters for performance + dependency isolation."""
    trip_wire = {"called": False}

    def trap(*args, **kwargs):
        trip_wire["called"] = True
        raise AssertionError("build_graph must not be called when build_graph=False")

    monkeypatch.setattr("ragkit.core.graph.builder.build_graph", trap)

    result = index_file(sample_txt, kb_name="kb", build_graph=False)
    assert trip_wire["called"] is False
    assert "graph_entities" not in result


# ===========================================================================
# Scenario E — re-index protection (option B: --replace flag + warning)
# ===========================================================================


def test_count_existing_chunks_returns_zero_when_index_missing(fake_es):
    """No index → no chunks to count, no error."""
    fake_es.es.indices.exists.return_value = False
    assert _count_existing_chunks_for_file("kb", "report.pdf") == 0


def test_count_existing_chunks_returns_es_count(fake_es):
    """When the index has matching docs, return the ES count value."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.count.return_value = {"count": 7}

    n = _count_existing_chunks_for_file("kb", "report.pdf")

    assert n == 7
    # Verify we actually queried by docnm_kwd:
    call = fake_es.es.count.call_args
    assert call.kwargs["index"] == "kb"
    assert call.kwargs["query"] == {"term": {"docnm_kwd": "report.pdf"}}


def test_count_existing_chunks_degrades_to_zero_on_es_error(fake_es):
    """ES failure must NOT crash indexing — degrade to 0 (treat as new file)."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.count.side_effect = RuntimeError("ES transient")

    assert _count_existing_chunks_for_file("kb", "report.pdf") == 0


def test_delete_existing_chunks_no_op_when_index_missing(fake_es):
    """No index → nothing to delete."""
    fake_es.es.indices.exists.return_value = False

    assert _delete_existing_chunks_for_file("kb", "report.pdf") == 0
    fake_es.es.delete_by_query.assert_not_called()


def test_delete_existing_chunks_calls_delete_by_query(fake_es):
    """When the index exists, issue delete_by_query and return the count."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.delete_by_query.return_value = {"deleted": 25}

    n = _delete_existing_chunks_for_file("kb", "report.pdf")

    assert n == 25
    call = fake_es.es.delete_by_query.call_args
    assert call.kwargs["index"] == "kb"
    assert call.kwargs["query"] == {"term": {"docnm_kwd": "report.pdf"}}
    assert call.kwargs["refresh"] is True


def test_delete_existing_chunks_raises_on_es_error(fake_es):
    """An ES failure must HARD ABORT — leaving stale + new chunks side-by-side
    would defeat the purpose of --replace."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.delete_by_query.side_effect = RuntimeError("ES transient")

    with pytest.raises(RuntimeError, match="Failed to delete stale chunks"):
        _delete_existing_chunks_for_file("kb", "report.pdf")


def test_index_file_default_warns_on_stale_chunks(
    sample_txt, fake_openai, fake_es, monkeypatch, capsys
):
    """Default behavior: detect existing chunks, warn, but APPEND new ones."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.count.return_value = {"count": 12}  # 12 existing chunks

    result = index_file(sample_txt, kb_name="kb", build_graph=False)

    # Warning surfaced via observe → rich console → captured stdout
    captured = capsys.readouterr()
    assert "already has 12 chunk" in captured.out
    assert "--replace" in captured.out
    # Delete should NOT have been called
    fake_es.es.delete_by_query.assert_not_called()
    # Indexing still proceeded
    assert result["replaced"] == 0
    assert result["chunks"] > 0


def test_index_file_replace_deletes_then_indexes(
    sample_txt, fake_openai, fake_es, monkeypatch, capsys
):
    """With --replace: count + delete + then index normally."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.count.return_value = {"count": 12}
    fake_es.es.delete_by_query.return_value = {"deleted": 12}

    result = index_file(sample_txt, kb_name="kb", build_graph=False, replace=True)

    # Confirmation surfaced via observe
    captured = capsys.readouterr()
    assert "Deleted 12 stale chunk" in captured.out
    # Delete WAS called
    fake_es.es.delete_by_query.assert_called_once()
    # Result reports the deletion
    assert result["replaced"] == 12
    assert result["chunks"] > 0


def test_index_file_replace_is_no_op_when_no_existing_chunks(
    sample_txt, fake_openai, fake_es
):
    """--replace on a fresh file: no delete needed, just index normally."""
    fake_es.es.indices.exists.return_value = False  # no index yet

    result = index_file(sample_txt, kb_name="kb", build_graph=False, replace=True)

    fake_es.es.delete_by_query.assert_not_called()
    assert result["replaced"] == 0
    assert result["chunks"] > 0


def test_index_file_no_warning_when_kb_is_fresh(
    sample_txt, fake_openai, fake_es, capsys
):
    """First-time index (no existing chunks) → no warning, no noise."""
    fake_es.es.indices.exists.return_value = False  # fresh KB

    result = index_file(sample_txt, kb_name="kb", build_graph=False)

    captured = capsys.readouterr()
    assert "already has" not in captured.out
    assert "stale" not in captured.out
    assert result["replaced"] == 0
