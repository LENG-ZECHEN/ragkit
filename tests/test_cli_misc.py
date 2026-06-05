"""CLI tests for the previously-untested commands and flags.

Covers: rag retrieve, rag kb info, rag index --recursive / --build-graph,
rag graph build / show --depth / clear (confirmation), rag graph info happy path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragkit.cli.app import app
from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Community, Entity, Relation
from ragkit.core.retriever import RetrievedChunk

runner = CliRunner()


# ============================================================
# rag retrieve
# ============================================================


def test_retrieve_prints_chunks(fake_openai, fake_es, monkeypatch):
    """`rag retrieve` must print every returned chunk to the console."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, top_k=5: [
            RetrievedChunk(
                rank=1, document_id="d1", document_name="alpha.pdf",
                content="DETECTABLE_ALPHA_TEXT", similarity=0.9,
                vector_similarity=0.9, term_similarity=0.9,
            ),
            RetrievedChunk(
                rank=2, document_id="d2", document_name="beta.pdf",
                content="DETECTABLE_BETA_TEXT", similarity=0.7,
                vector_similarity=0.7, term_similarity=0.7,
            ),
        ],
    )
    result = runner.invoke(app, ["retrieve", "Q", "--kb", "kb"])
    assert result.exit_code == 0
    assert "DETECTABLE_ALPHA_TEXT" in result.stdout
    assert "DETECTABLE_BETA_TEXT" in result.stdout
    assert "alpha.pdf" in result.stdout
    assert "beta.pdf" in result.stdout


def test_retrieve_warns_on_zero_results(fake_openai, fake_es, monkeypatch):
    """No matches → user-visible warning, not silent success."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, top_k=5: [],
    )
    result = runner.invoke(app, ["retrieve", "Q", "--kb", "kb"])
    assert result.exit_code == 0
    assert "No matches" in result.stdout


def test_retrieve_top_k_threaded(fake_openai, fake_es, monkeypatch):
    """--top-k must reach the underlying retriever."""
    captured = {}

    def fake(question, kb_name, top_k=5):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr("ragkit.core.retriever.retrieve", fake)
    result = runner.invoke(app, ["retrieve", "Q", "--kb", "kb", "--top-k", "11"])
    assert result.exit_code == 0
    assert captured["top_k"] == 11


# ============================================================
# rag kb info
# ============================================================


def test_kb_info_displays_stats_and_documents(fake_es, monkeypatch):
    """kb info on a populated KB must show document and chunk counts."""
    from ragkit.core.kb_manager import KbInfo

    monkeypatch.setattr(
        "ragkit.core.kb_manager.kb_info",
        lambda name: KbInfo(name=name, document_count=3, chunk_count=42),
    )
    monkeypatch.setattr(
        "ragkit.core.kb_manager.kb_documents",
        lambda name: [
            {"document_name": "alpha.pdf", "chunks": 25},
            {"document_name": "beta.docx", "chunks": 17},
        ],
    )

    result = runner.invoke(app, ["kb", "info", "finance"])
    assert result.exit_code == 0
    assert "finance" in result.stdout
    assert "3" in result.stdout  # documents count
    assert "42" in result.stdout  # chunks count
    assert "alpha.pdf" in result.stdout
    assert "beta.docx" in result.stdout


# ============================================================
# rag index --recursive / --build-graph
# ============================================================


def test_index_recursive_picks_up_nested_files(tmp_path, fake_openai, fake_es, monkeypatch):
    """--recursive must descend into subdirectories."""
    # Build a small tree: root/a.txt, root/sub/b.txt
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta", encoding="utf-8")

    indexed: list[str] = []

    def fake_index(path, kb_name, *, build_graph=False, progress_cb=None):
        indexed.append(Path(path).name)
        return {"file": Path(path).name, "chunks": 1, "kb": kb_name}

    monkeypatch.setattr("ragkit.core.indexer.index_file", fake_index)

    # Without --recursive: only the top-level a.txt
    result = runner.invoke(app, ["index", str(tmp_path), "--kb", "kb"])
    assert result.exit_code == 0
    assert "a.txt" in indexed
    assert "b.txt" not in indexed

    # With --recursive: both
    indexed.clear()
    result = runner.invoke(app, ["index", str(tmp_path), "--kb", "kb", "--recursive"])
    assert result.exit_code == 0
    assert "a.txt" in indexed
    assert "b.txt" in indexed


def test_index_build_graph_flag_passes_through(tmp_path, fake_openai, fake_es, monkeypatch):
    """--build-graph must reach index_file as build_graph=True."""
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")

    captured: dict = {}

    def fake_index(path, kb_name, *, build_graph=False, progress_cb=None):
        captured["build_graph"] = build_graph
        return {"file": Path(path).name, "chunks": 1, "kb": kb_name}

    monkeypatch.setattr("ragkit.core.indexer.index_file", fake_index)

    # Default: False
    runner.invoke(app, ["index", str(tmp_path / "doc.txt"), "--kb", "k"])
    assert captured["build_graph"] is False

    # Flag set: True
    runner.invoke(app, ["index", str(tmp_path / "doc.txt"), "--kb", "k", "--build-graph"])
    assert captured["build_graph"] is True


# ============================================================
# rag graph build
# ============================================================


def test_graph_build_errors_when_kb_missing(fake_es):
    """`rag graph build --kb X` for a non-existent KB exits non-zero with
    a helpful message — not a stack trace."""
    fake_es.es.indices.exists.return_value = False
    result = runner.invoke(app, ["graph", "build", "--kb", "ghost-kb"])
    assert result.exit_code != 0
    assert "ghost-kb" in result.stdout
    assert "does not exist" in result.stdout.lower() or "index" in result.stdout.lower()


def test_graph_build_errors_on_empty_kb_name(fake_es):
    """Whitespace-only --kb must reject. ISS-003 now catches this at
    validate_kb_name (raised as ValueError before reaching graph_build)."""
    result = runner.invoke(app, ["graph", "build", "--kb", "   "])
    assert result.exit_code != 0
    # New error path raises ValueError("Invalid kb name ...") via kb_validator.
    exc = result.exception
    assert isinstance(exc, ValueError) and "Invalid kb name" in str(exc)


def test_graph_build_warns_when_no_chunks(fake_es):
    """KB exists but is empty → useful warning, not a crash."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {"hits": {"hits": []}, "_scroll_id": "sid"}
    fake_es.es.scroll.return_value = {"hits": {"hits": []}, "_scroll_id": "sid"}

    result = runner.invoke(app, ["graph", "build", "--kb", "kb"])
    assert result.exit_code != 0
    assert "No chunks" in result.stdout


def test_graph_build_passes_summarize_flag(tmp_path, fake_openai, fake_es, monkeypatch):
    """--no-summarize must reach the build_graph function."""
    fake_es.es.indices.exists.return_value = True
    # One hit, then empty so the scroll loop exits.
    fake_es.es.search.return_value = {
        "hits": {"hits": [{"_id": "c1", "_source": {"content_with_weight": "Alice met Bob."}}]},
        "_scroll_id": "sid",
    }
    fake_es.es.scroll.return_value = {"hits": {"hits": []}, "_scroll_id": "sid"}

    captured: dict = {}

    def fake_build(chunks, kb_name, *, summarize, max_summary_communities,
                   consolidate_descriptions=True, max_consolidation_calls=20,
                   progress_cb=None, **kw):
        # ISS-011: also capture consolidation kwargs so the test catches a
        # broken --no-consolidate / --max-consolidations flag mapping.
        captured["summarize"] = summarize
        captured["max"] = max_summary_communities
        captured["consolidate"] = consolidate_descriptions
        captured["max_consolidations"] = max_consolidation_calls
        store = NetworkXGraphStore(path=tmp_path / "g.json")
        store.upsert_entity(Entity(name="alice", type="person"))
        return store

    monkeypatch.setattr("ragkit.core.graph.builder.build_graph", fake_build)

    result = runner.invoke(app, ["graph", "build", "--kb", "kb", "--no-summarize"])
    assert result.exit_code == 0
    assert captured["summarize"] is False

    runner.invoke(app, ["graph", "build", "--kb", "kb", "--max-summaries", "7"])
    assert captured["max"] == 7

    # ISS-011: --no-consolidate and --max-consolidations propagate
    runner.invoke(app, ["graph", "build", "--kb", "kb", "--no-consolidate"])
    assert captured["consolidate"] is False

    runner.invoke(app, ["graph", "build", "--kb", "kb", "--max-consolidations", "5"])
    assert captured["max_consolidations"] == 5


def test_graph_build_debug_flag_enables_observe(tmp_path, fake_openai, fake_es, monkeypatch):
    """ISS-011: `rag graph build --debug` must call observe.enable_debug()
    so the trace_xxx() calls inside the build pipeline actually emit output."""
    from ragkit.cli import observe

    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {
        "hits": {"hits": [{"_id": "c1", "_source": {"content_with_weight": "x"}}]},
        "_scroll_id": "sid",
    }
    fake_es.es.scroll.return_value = {"hits": {"hits": []}, "_scroll_id": "sid"}

    def fake_build(chunks, kb_name, **kw):
        store = NetworkXGraphStore(path=tmp_path / "g.json")
        store.upsert_entity(Entity(name="x", type="t"))
        return store

    monkeypatch.setattr("ragkit.core.graph.builder.build_graph", fake_build)
    # Always start from a clean state — observe is module-global.
    observe.disable_debug()
    try:
        result = runner.invoke(app, ["graph", "build", "--kb", "kb", "--debug"])
        assert result.exit_code == 0
        assert observe.is_debug() is True
    finally:
        observe.disable_debug()


# ============================================================
# rag graph info (happy path)
# ============================================================


def test_graph_info_displays_populated_graph(tmp_path, monkeypatch):
    """`rag graph info` on a non-empty graph shows entity/relation/community
    counts and per-type breakdown."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))

    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    store = NetworkXGraphStore(path=path)
    store.upsert_entity(Entity(name="alpha", type="person"))
    store.upsert_entity(Entity(name="beta", type="person"))
    store.upsert_entity(Entity(name="acme", type="organization"))
    store.upsert_relation(Relation(source="alpha", target="acme"))
    store.set_communities([Community(id=0, entity_names=["alpha", "beta", "acme"])])
    store.save()

    result = runner.invoke(app, ["graph", "info", "kb"])
    assert result.exit_code == 0
    # Counts (3 entities, 1 relation, 1 community)
    assert "3" in result.stdout
    # Type breakdown rows
    assert "person" in result.stdout
    assert "organization" in result.stdout


# ============================================================
# rag graph show --depth
# ============================================================


def test_graph_show_respects_depth_flag(tmp_path, monkeypatch):
    """--depth 2 reaches grandchildren that --depth 1 misses."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    store = NetworkXGraphStore(path=path)
    # chain: a — b — c — d
    store.upsert_entity(Entity(name="a", type="t"))
    store.upsert_entity(Entity(name="b", type="t"))
    store.upsert_entity(Entity(name="c", type="t"))
    store.upsert_entity(Entity(name="d_node", type="t"))
    store.upsert_relation(Relation(source="a", target="b"))
    store.upsert_relation(Relation(source="b", target="c"))
    store.upsert_relation(Relation(source="c", target="d_node"))
    store.save()

    r1 = runner.invoke(app, ["graph", "show", "kb", "a", "--depth", "1"])
    assert r1.exit_code == 0
    assert "b" in r1.stdout
    assert "d_node" not in r1.stdout  # depth=1 should not reach d

    r3 = runner.invoke(app, ["graph", "show", "kb", "a", "--depth", "3"])
    assert r3.exit_code == 0
    assert "d_node" in r3.stdout  # depth=3 should reach d


# ============================================================
# rag graph clear (confirmation path)
# ============================================================


def test_graph_report_shows_structured_fields(tmp_path, monkeypatch):
    """`rag graph report KB ID` prints title/summary/rank/findings."""
    from ragkit.core.graph.types import Finding

    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    store = NetworkXGraphStore(path=path)
    store.upsert_entity(Entity(name="x", type="t"))
    store.set_communities([
        Community(
            id=7,
            entity_names=["x"],
            title="THE_TITLE_MARKER",
            summary="THE_SUMMARY_MARKER",
            rank=6.5,
            rank_explanation="THE_RANK_REASON",
            findings=[
                Finding(summary="FIND_SUMMARY_1", explanation="FIND_EXPLAIN_1"),
            ],
        )
    ])
    store.save()

    result = runner.invoke(app, ["graph", "report", "kb", "7"])
    assert result.exit_code == 0
    assert "THE_TITLE_MARKER" in result.stdout
    assert "THE_SUMMARY_MARKER" in result.stdout
    assert "THE_RANK_REASON" in result.stdout
    assert "FIND_SUMMARY_1" in result.stdout
    assert "FIND_EXPLAIN_1" in result.stdout


def test_graph_report_errors_when_id_not_found(tmp_path, monkeypatch):
    """Unknown community ID must exit non-zero, not crash."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "entities": [], "relations": [], "communities": []
    }))
    result = runner.invoke(app, ["graph", "report", "kb", "999"])
    assert result.exit_code != 0


def test_graph_clear_aborts_when_user_declines(tmp_path, monkeypatch):
    """Without --yes, typing 'n' must NOT delete the graph file."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entities": [], "relations": [], "communities": []}))

    result = runner.invoke(app, ["graph", "clear", "kb"], input="n\n")
    assert result.exit_code == 0
    assert path.exists()  # not deleted


def test_graph_clear_proceeds_when_user_accepts(tmp_path, monkeypatch):
    """Confirmation 'y' deletes the graph file."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entities": [], "relations": [], "communities": []}))

    result = runner.invoke(app, ["graph", "clear", "kb"], input="y\n")
    assert result.exit_code == 0
    assert not path.exists()


# ============================================================
# rag kb empty list message
# ============================================================


def test_kb_list_empty_shows_helpful_hint(fake_es):
    """An empty KB list should hint at `rag index`, not just print nothing."""
    fake_es.list_indices.return_value = []
    result = runner.invoke(app, ["kb", "list"])
    assert result.exit_code == 0
    assert "index" in result.stdout.lower() or "no knowledge bases" in result.stdout.lower()
