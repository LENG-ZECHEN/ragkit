"""Graph builder pipeline + CLI graph commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ragkit.cli.app import app
from ragkit.core.graph.builder import build_graph
from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.types import Entity, Relation

runner = CliRunner()


# ----- builder end-to-end -------------------------------------------------


def test_build_graph_with_no_chunks_returns_empty_store(tmp_path, fake_openai):
    """Edge case: indexer built an empty graph (no chunks) → don't crash."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    out = build_graph([], kb_name="t", summarize=False, store=store)
    assert out.entity_count() == 0


def test_build_graph_aggregates_across_chunks(tmp_path, fake_openai):
    """Same entity appearing in two chunks should be merged into one node."""
    # Script the LLM to emit the same entity in both chunks.
    extraction = json.dumps({
        "entities": [{"name": "Qwen", "type": "model", "description": "An LLM."}],
        "relations": [],
    })
    fake_openai.chat_script = [("content", extraction)]

    store = NetworkXGraphStore(path=tmp_path / "g.json")
    chunks = [
        {"id": "c1", "content_with_weight": "Qwen is a model."},
        {"id": "c2", "content_with_weight": "Qwen is widely used."},
    ]
    build_graph(chunks, kb_name="t", summarize=False, store=store)

    # One merged entity, with both chunks as sources.
    assert store.entity_count() == 1
    qwen = store.get_entity("qwen")
    assert qwen is not None
    assert set(qwen.source_chunks) == {"c1", "c2"}


def test_build_graph_skips_summarize_when_no_communities(tmp_path, fake_openai):
    """When extraction legitimately produces nothing (e.g. empty/whitespace
    chunks) the builder should not call the summarizer and should not raise."""
    # Empty text legitimately produces zero entities — no LLM call is made.
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    build_graph(
        [{"id": "c1", "content_with_weight": ""}],
        kb_name="t",
        summarize=True,
        store=store,
    )
    assert store.all_communities() == []


def test_build_graph_progress_callback_invoked(tmp_path, fake_openai):
    """The CLI's progress bar needs extracting + clustering signals.
    (summarizing only fires when there are summaries to build.)"""
    extraction = json.dumps({
        "entities": [
            {"name": "x", "type": "t", "description": ""},
            {"name": "y", "type": "t", "description": ""},
        ],
        "relations": [{"source": "x", "target": "y", "description": "r"}],
    })
    fake_openai.chat_script = [("content", extraction)]

    seen_stages: set[str] = set()

    def cb(stage: str, current: int, total: int) -> None:
        seen_stages.add(stage)
        assert current <= total
        assert total > 0

    store = NetworkXGraphStore(path=tmp_path / "g.json")
    build_graph(
        [{"id": "c1", "content_with_weight": "x and y"}],
        kb_name="t",
        summarize=False,
        progress_cb=cb,
        store=store,
    )
    assert "extracting" in seen_stages
    assert "clustering" in seen_stages


def test_build_graph_aborts_when_all_extractions_fail(tmp_path, fake_openai, monkeypatch):
    """If every extraction fails (e.g. quota exhausted), refuse to save an
    empty graph that would mask the real problem."""
    def boom(**kwargs):
        raise RuntimeError("Connection refused")

    monkeypatch.setattr(fake_openai.chat.completions, "create", boom)

    store = NetworkXGraphStore(path=tmp_path / "g.json")
    with pytest.raises(RuntimeError, match="extraction failed"):
        build_graph(
            [{"id": f"c{i}", "content_with_weight": "real text"} for i in range(5)],
            kb_name="t",
            summarize=False,
            store=store,
        )


def test_summarizer_preserves_all_communities_above_cap(tmp_path, fake_openai):
    """Regression test for the critical data-loss bug: max_communities=N must
    NOT delete communities beyond N from the store."""
    from ragkit.core.graph.summarizer import summarize_all
    from ragkit.core.graph.types import Community, Relation

    fake_openai.chat_script = [("content", "A short summary.")]

    store = NetworkXGraphStore(path=tmp_path / "g.json")
    store.upsert_relation(Relation(source="a", target="b"))
    store.upsert_relation(Relation(source="c", target="d"))
    store.upsert_relation(Relation(source="e", target="f"))
    store.set_communities([
        Community(id=0, entity_names=["a", "b"]),
        Community(id=1, entity_names=["c", "d"]),
        Community(id=2, entity_names=["e", "f"]),
    ])

    summarize_all(store, max_communities=1)

    # All three communities still present (only the first got a summary).
    saved = store.all_communities()
    assert len(saved) == 3
    assert saved[0].summary  # got summarized
    # The tail communities are still in the store (would have been deleted by the bug).
    assert saved[2].entity_names == ["e", "f"]


def test_summarizer_handles_llm_failure_marks_empty(tmp_path, fake_openai, monkeypatch):
    """LLM error during summarization → empty string for that community,
    other communities still get summarized correctly."""
    from ragkit.core.graph.summarizer import summarize_all
    from ragkit.core.graph.types import Community, Relation

    def boom(**kwargs):
        raise RuntimeError("API down")

    monkeypatch.setattr(fake_openai.chat.completions, "create", boom)

    store = NetworkXGraphStore(path=tmp_path / "g.json")
    store.upsert_relation(Relation(source="a", target="b"))
    store.set_communities([Community(id=0, entity_names=["a", "b"])])

    failures = summarize_all(store)
    assert failures == 1
    assert store.all_communities()[0].summary == ""


def test_build_graph_runs_consolidation_by_default(tmp_path, fake_openai, monkeypatch):
    """build_graph must call consolidate_all by default."""
    from ragkit.core.graph.description_merger import ConsolidationResult

    called = {"n": 0}

    def fake_consolidate(store, **kw):
        called["n"] += 1
        return ConsolidationResult()

    monkeypatch.setattr("ragkit.core.graph.builder.consolidate_all", fake_consolidate)

    extraction = json.dumps({
        "entities": [{"name": "x", "type": "t", "description": ""}],
        "relations": [],
    })
    fake_openai.chat_script = [("content", extraction)]

    store = NetworkXGraphStore(path=tmp_path / "g.json")
    build_graph(
        [{"id": "c1", "content_with_weight": "anything"}],
        kb_name="t",
        summarize=False,
        store=store,
    )
    assert called["n"] == 1


def test_build_graph_skips_consolidation_when_flag_off(tmp_path, fake_openai, monkeypatch):
    """--no-consolidate path: consolidate_all is NOT called."""
    called = {"n": 0}

    def trap(store, **kw):
        called["n"] += 1

    monkeypatch.setattr("ragkit.core.graph.builder.consolidate_all", trap)

    extraction = json.dumps({
        "entities": [{"name": "x", "type": "t", "description": ""}],
        "relations": [],
    })
    fake_openai.chat_script = [("content", extraction)]

    store = NetworkXGraphStore(path=tmp_path / "g.json")
    build_graph(
        [{"id": "c1", "content_with_weight": "anything"}],
        kb_name="t",
        summarize=False,
        consolidate_descriptions=False,
        store=store,
    )
    assert called["n"] == 0


def test_build_graph_persists_to_disk(tmp_path, fake_openai):
    """Build → file must exist & be reloadable."""
    extraction = json.dumps({
        "entities": [{"name": "alpha", "type": "concept", "description": "α"}],
        "relations": [],
    })
    fake_openai.chat_script = [("content", extraction)]

    path = tmp_path / "g.json"
    store = NetworkXGraphStore(path=path)
    build_graph(
        [{"id": "c1", "content_with_weight": "Alpha is here."}],
        kb_name="t",
        summarize=False,
        store=store,
    )
    assert path.exists()
    reloaded = NetworkXGraphStore(path=path)
    assert reloaded.get_entity("alpha") is not None


# ----- CLI graph subcommands ---------------------------------------------


def test_graph_info_warns_when_no_graph(tmp_path, monkeypatch):
    """If user runs `graph info` on a KB that has no graph, show a helpful message."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    result = runner.invoke(app, ["graph", "info", "no-such-kb"])
    assert result.exit_code == 0
    assert "no graph" in result.stdout.lower() or "build" in result.stdout.lower()


def test_graph_show_errors_on_unknown_entity(tmp_path, monkeypatch):
    """`graph show <kb> <entity>` for a non-existent entity must exit non-zero."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))
    # Create empty graph file so the store opens cleanly.
    (tmp_path / "graphs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "graphs" / "kb.json").write_text(
        json.dumps({"entities": [], "relations": [], "communities": []})
    )
    result = runner.invoke(app, ["graph", "show", "kb", "ghost"])
    assert result.exit_code != 0


def test_graph_show_displays_entity_neighbors(tmp_path, monkeypatch):
    """Happy path: entity exists → details + neighbors printed."""
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))

    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    store = NetworkXGraphStore(path=path)
    store.upsert_entity(Entity(name="qwen", type="model", description="An LLM."))
    store.upsert_entity(Entity(name="alibaba", type="org", description="A company."))
    store.upsert_relation(Relation(source="qwen", target="alibaba", description="made by"))
    store.save()

    result = runner.invoke(app, ["graph", "show", "kb", "qwen"])
    assert result.exit_code == 0
    assert "qwen" in result.stdout.lower()
    assert "alibaba" in result.stdout.lower()


def test_graph_clear_with_yes_deletes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path))

    path = tmp_path / "graphs" / "kb.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")

    result = runner.invoke(app, ["graph", "clear", "kb", "--yes"])
    assert result.exit_code == 0
    assert not path.exists()
