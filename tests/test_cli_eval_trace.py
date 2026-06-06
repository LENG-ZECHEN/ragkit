"""CLI tests for --eval-trace, --eval-out, --param on ``rag ask`` / ``rag retrieve``.

These guard the Phase 0 evaluation instrumentation layer:
  * --eval-trace prints a complete EvalTrace JSON object
  * --eval-out PATH writes it to a file instead of stdout
  * --param key=value layers overrides into the trace's params
  * Without any of these flags, behavior must be unchanged (byte-for-byte
    same stdout as today's CLI).
"""

from __future__ import annotations

import json
from pathlib import Path

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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _extract_trace_json(stdout: str) -> dict:
    """Pull the eval-trace JSON object out of mixed CLI stdout.

    The trace JSON is printed last via plain ``print()`` after the rendered
    answer. We scan from the LAST '{' that starts a balanced JSON object.
    """
    # Find the LAST balanced JSON object.
    depth = 0
    end = -1
    last_start = -1
    for i in range(len(stdout) - 1, -1, -1):
        ch = stdout[i]
        if ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0 and end != -1:
                last_start = i
                break
    assert last_start != -1 and end != -1, (
        f"No JSON object found in stdout: {stdout!r}"
    )
    return json.loads(stdout[last_start : end + 1])


# --------------------------------------------------------------------------
# Trace schema completeness
# --------------------------------------------------------------------------


def test_ask_eval_trace_emits_complete_json(fake_openai, fake_es, monkeypatch):
    """--eval-trace must print a JSON object with every documented top-level key."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )
    fake_openai.chat_script = [("content", "answer text")]

    result = runner.invoke(app, [
        "ask", "What is X?", "--kb", "kb",
        "--eval-trace",
    ])
    assert result.exit_code == 0, result.stdout
    trace = _extract_trace_json(result.stdout)

    for key in (
        "question", "kb", "mode", "top_k", "level", "timestamp_iso",
        "retrieved", "timing", "cost", "params", "answer",
    ):
        assert key in trace, f"missing key {key} in trace: {trace}"

    assert trace["question"] == "What is X?"
    assert trace["kb"] == "kb"
    assert trace["mode"] == "vector"
    assert trace["top_k"] == 5
    assert isinstance(trace["retrieved"], list)
    assert isinstance(trace["timing"], dict)
    assert isinstance(trace["cost"], dict)
    assert isinstance(trace["params"], dict)


def test_ask_eval_trace_carries_param_overrides_in_params(fake_openai, fake_es, monkeypatch):
    """--param vector_similarity_weight=0.5 must surface as 0.5 in trace.params."""
    captured = {}

    def fake_retrieve(question, kb_name, *, top_k, **kw):
        # The retriever reads the override via eval_context.get() — confirm it
        # actually saw 0.5.
        from ragkit import eval_context
        captured["vsw"] = eval_context.get("vector_similarity_weight", 0.6)
        return [_one_chunk()]

    monkeypatch.setattr("ragkit.core.retriever.retrieve", fake_retrieve)
    fake_openai.chat_script = [("content", "ans")]

    result = runner.invoke(app, [
        "ask", "Q", "--kb", "kb",
        "--eval-trace",
        "--param", "vector_similarity_weight=0.5",
    ])
    assert result.exit_code == 0, result.stdout
    trace = _extract_trace_json(result.stdout)

    assert trace["params"]["vector_similarity_weight"] == 0.5
    assert captured["vsw"] == 0.5


def test_ask_eval_trace_multiple_params(fake_openai, fake_es, monkeypatch):
    """Multiple --param flags must all apply (repeatable option)."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )
    fake_openai.chat_script = [("content", "ans")]

    result = runner.invoke(app, [
        "ask", "Q", "--kb", "kb", "--eval-trace",
        "--param", "vector_similarity_weight=0.3",
        "--param", "similarity_threshold=0.05",
    ])
    assert result.exit_code == 0, result.stdout
    trace = _extract_trace_json(result.stdout)

    assert trace["params"]["vector_similarity_weight"] == 0.3
    assert trace["params"]["similarity_threshold"] == 0.05


def test_ask_eval_trace_defaults_appear_in_params(fake_openai, fake_es, monkeypatch):
    """Non-overridden params still appear in trace.params at their default values."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )
    fake_openai.chat_script = [("content", "ans")]

    result = runner.invoke(app, ["ask", "Q", "--kb", "kb", "--eval-trace"])
    assert result.exit_code == 0
    trace = _extract_trace_json(result.stdout)

    # All known params should be present.
    expected_keys = {
        "vector_similarity_weight", "similarity_threshold", "chunk_token_num",
        "local_top_k_seeds", "local_top_k_text_units", "local_top_k_communities",
        "local_top_k_entities", "local_top_k_relations", "global_top_k_reports",
        "map_batch_token_budget", "rating_threshold", "default_final_top_k",
    }
    assert expected_keys.issubset(trace["params"].keys())
    # Default value sanity check.
    assert trace["params"]["vector_similarity_weight"] == 0.6


# --------------------------------------------------------------------------
# --eval-out file output
# --------------------------------------------------------------------------


def test_ask_eval_out_writes_to_file(fake_openai, fake_es, monkeypatch, tmp_path):
    """--eval-out PATH writes the trace there (not stdout)."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )
    fake_openai.chat_script = [("content", "ans")]

    out_file = tmp_path / "trace.json"
    result = runner.invoke(app, [
        "ask", "Q", "--kb", "kb",
        "--eval-trace",
        "--eval-out", str(out_file),
    ])
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload["question"] == "Q"
    assert payload["mode"] == "vector"


# --------------------------------------------------------------------------
# Bad --param input
# --------------------------------------------------------------------------


def test_ask_bad_param_exits_nonzero(fake_openai, fake_es):
    """An unknown --param key must exit non-zero with a clear message."""
    result = runner.invoke(app, [
        "ask", "Q", "--kb", "kb",
        "--param", "foo=bar",
    ])
    assert result.exit_code != 0


def test_ask_bad_param_syntax_exits_nonzero(fake_openai, fake_es):
    """--param missing '=' must exit non-zero."""
    result = runner.invoke(app, [
        "ask", "Q", "--kb", "kb",
        "--param", "noequalshere",
    ])
    assert result.exit_code != 0


# --------------------------------------------------------------------------
# No-flag invariant — default behavior unchanged
# --------------------------------------------------------------------------


def test_ask_without_eval_flags_no_json_dump(fake_openai, fake_es, monkeypatch):
    """When neither --eval-trace nor --eval-out is set, no JSON trace appears
    after the rendered answer. This guards the byte-for-byte-identical
    invariant for the common path."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )
    fake_openai.chat_script = [("content", "the answer")]

    result = runner.invoke(app, ["ask", "Q", "--kb", "kb"])
    assert result.exit_code == 0
    # The answer is there.
    assert "the answer" in result.stdout
    # But no trace JSON.  A trace would contain "timestamp_iso" which is unique
    # to the EvalTrace schema — assert that marker is absent.
    assert "timestamp_iso" not in result.stdout
    # And no params field either.
    assert '"params"' not in result.stdout


# --------------------------------------------------------------------------
# retrieve subcommand
# --------------------------------------------------------------------------


def test_retrieve_eval_trace_emits_json(fake_openai, fake_es, monkeypatch):
    """rag retrieve --eval-trace must also emit a complete trace (mode=vector,
    answer=None)."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )

    result = runner.invoke(app, [
        "retrieve", "Q", "--kb", "kb",
        "--eval-trace",
    ])
    assert result.exit_code == 0, result.stdout
    trace = _extract_trace_json(result.stdout)
    assert trace["mode"] == "vector"
    assert trace["answer"] is None  # retrieve runs no generator
    assert trace["question"] == "Q"


def test_retrieve_eval_trace_with_param(fake_openai, fake_es, monkeypatch):
    """rag retrieve --param must layer into trace.params too."""
    monkeypatch.setattr(
        "ragkit.core.retriever.retrieve",
        lambda question, kb_name, **kw: [_one_chunk()],
    )

    result = runner.invoke(app, [
        "retrieve", "Q", "--kb", "kb", "--eval-trace",
        "--param", "vector_similarity_weight=0.42",
    ])
    assert result.exit_code == 0
    trace = _extract_trace_json(result.stdout)
    assert trace["params"]["vector_similarity_weight"] == 0.42
