"""CLI tests for `rag ask` retrieval modes + output flags.

The audit found these flags were wired but never exercised through the CLI:
  --mode {vector|local|global|hybrid|<invalid>}
  --top-k
  --thinking
  --json
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ragkit.cli.app import app
from ragkit.core.retriever import RetrievedChunk

runner = CliRunner()


def _one_chunk(rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        document_id=f"d{rank}",
        document_name="doc.pdf",
        content="some retrieved content",
        similarity=0.85,
        vector_similarity=0.85,
        term_similarity=0.85,
    )


# ----- mode routing ------------------------------------------------------


def test_ask_mode_vector_calls_vector_retriever(fake_openai, fake_es, monkeypatch):
    """--mode vector (the default) must call the BM25+dense retriever, NOT
    the graph functions."""
    called = {"vector": False, "local": False, "global": False, "hybrid": False}

    def vec(question, kb_name, **kw):
        called["vector"] = True
        return [_one_chunk()]

    monkeypatch.setattr("ragkit.core.retriever.retrieve", vec)
    # Trip-wires: graph retrievers must NOT be invoked.
    monkeypatch.setattr(
        "ragkit.core.graph.retriever.retrieve_local",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("local should not run")),
    )

    fake_openai.chat_script = [("content", "ok")]

    result = runner.invoke(app, ["ask", "Q", "--kb", "k", "--mode", "vector"])
    assert result.exit_code == 0
    assert called["vector"] is True


def test_ask_mode_local_calls_graph_local(fake_openai, monkeypatch):
    """--mode local routes to graph.retrieve_local, not vector retrieve."""
    captured = {}

    def fake_local(question, kb_name, **kw):
        captured["question"] = question
        captured["kb"] = kb_name
        from ragkit.core.graph.retriever import GraphHit
        return [GraphHit(rank=1, kind="entity", title="X", content="x", extra={})]

    monkeypatch.setattr("ragkit.core.graph.retriever.retrieve_local", fake_local)
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("vector should not run")),
    )
    fake_openai.chat_script = [("content", "ok")]

    result = runner.invoke(app, ["ask", "About qwen", "-k", "kb", "--mode", "local"])
    assert result.exit_code == 0
    assert captured["question"] == "About qwen"
    assert captured["kb"] == "kb"


def test_ask_mode_global_calls_graph_global(fake_openai, monkeypatch):
    called = {"global": False}

    def fake_global(question, kb_name, **kw):
        called["global"] = True
        from ragkit.core.graph.retriever import GraphHit
        return [GraphHit(rank=1, kind="community", title="C0", content="summary", extra={})]

    monkeypatch.setattr("ragkit.core.graph.retriever.retrieve_global", fake_global)
    fake_openai.chat_script = [("content", "ok")]

    result = runner.invoke(app, ["ask", "Q", "-k", "kb", "--mode", "global"])
    assert result.exit_code == 0
    assert called["global"] is True


def test_ask_mode_hybrid_calls_graph_hybrid(fake_openai, monkeypatch):
    called = {"hybrid": False}

    def fake_hybrid(question, kb_name, **kw):
        called["hybrid"] = True
        from ragkit.core.graph.retriever import GraphHit
        return [GraphHit(rank=1, kind="chunk", title="d", content="c", extra={})]

    monkeypatch.setattr("ragkit.core.graph.retriever.retrieve_hybrid", fake_hybrid)
    fake_openai.chat_script = [("content", "ok")]

    result = runner.invoke(app, ["ask", "Q", "-k", "kb", "--mode", "hybrid"])
    assert result.exit_code == 0
    assert called["hybrid"] is True


def test_ask_rejects_invalid_mode(fake_openai, fake_es):
    """Unknown mode must exit with the usage-error code (2) and list valid modes."""
    result = runner.invoke(app, ["ask", "Q", "--mode", "telepathy"])
    assert result.exit_code == 2
    assert "telepathy" in result.stdout
    # Should hint at valid choices.
    assert "vector" in result.stdout and "hybrid" in result.stdout


# ----- top_k passthrough ------------------------------------------------


def test_ask_top_k_threaded_through_to_retriever(fake_openai, fake_es, monkeypatch):
    """--top-k 12 must reach the retriever (otherwise the flag does nothing)."""
    captured = {}

    def fake_retrieve(question, kb_name, *, top_k, **kw):
        captured["top_k"] = top_k
        return [_one_chunk()]

    monkeypatch.setattr("ragkit.core.retriever.retrieve", fake_retrieve)
    fake_openai.chat_script = [("content", "ok")]

    result = runner.invoke(app, ["ask", "Q", "--kb", "k", "--top-k", "12"])
    assert result.exit_code == 0
    assert captured["top_k"] == 12


# ----- --json output -----------------------------------------------------


def test_ask_json_emits_structured_payload(fake_openai, fake_es, monkeypatch):
    """--json must emit a single JSON object containing question/kb/answer/references."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )
    fake_openai.chat_script = [("content", "The answer.")]

    result = runner.invoke(app, ["ask", "What is X?", "--kb", "kb", "--json"])
    assert result.exit_code == 0

    # rich's print_json may pretty-print; extract the first JSON object span.
    out = result.stdout
    start = out.find("{")
    end = out.rfind("}")
    assert start != -1 and end != -1, f"No JSON object in output: {out!r}"
    payload = json.loads(out[start : end + 1])

    assert payload["question"] == "What is X?"
    assert payload["kb"] == "kb"
    assert payload["answer"] == "The answer."
    assert isinstance(payload["references"], list)
    assert len(payload["references"]) == 1
    assert payload["references"][0]["document_name"] == "doc.pdf"


# ----- --thinking passthrough -------------------------------------------


def test_ask_thinking_flag_renders_reasoning(fake_openai, fake_es, monkeypatch):
    """--thinking causes 'thinking' events to be printed; default hides them."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )
    fake_openai.chat_script = [
        ("thinking", "REASONING_TRACE_MARKER"),
        ("content", "Final answer."),
    ]

    without_thinking = runner.invoke(app, ["ask", "Q", "--kb", "kb"])
    assert without_thinking.exit_code == 0
    assert "REASONING_TRACE_MARKER" not in without_thinking.stdout
    assert "Final answer." in without_thinking.stdout

    with_thinking = runner.invoke(app, ["ask", "Q", "--kb", "kb", "--thinking"])
    assert with_thinking.exit_code == 0
    assert "REASONING_TRACE_MARKER" in with_thinking.stdout


# ----- retrieval failure --------------------------------------------------


def test_ask_exits_on_retrieval_failure(fake_openai, fake_es, monkeypatch):
    """A failing retrieve raises a Python exception that the CLI must convert
    to a clean non-zero exit (not a stacktrace) with an error line."""
    def broken(*args, **kwargs):
        raise RuntimeError("ES connection refused")

    monkeypatch.setattr("ragkit.core.retriever.retrieve", broken)

    result = runner.invoke(app, ["ask", "Q", "--kb", "k"])
    assert result.exit_code == 2
    assert "Retrieval failed" in result.stdout
