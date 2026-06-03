"""CLI integration — typer command parsing and error contracts.

These run the CLI with the typer test runner so we catch help text, arg
parsing, exit codes, and wiring between commands and core.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ragkit.cli.app import app

runner = CliRunner()


# ----- structure & help ---------------------------------------------------


def test_no_args_shows_help():
    """Running `rag` with nothing prints help text (typer's no_args_is_help
    exits with code 2, which is the conventional 'usage error' code)."""
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout or "Commands" in result.stdout


def test_top_level_commands_are_registered():
    """If we accidentally drop a command, this catches it immediately."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "ask", "chat", "retrieve", "doctor", "kb"):
        assert cmd in result.stdout


def test_kb_subcommands_are_registered():
    result = runner.invoke(app, ["kb", "--help"])
    assert result.exit_code == 0
    for sub in ("list", "info", "delete"):
        assert sub in result.stdout


# ----- input validation --------------------------------------------------


def test_index_rejects_missing_path():
    """typer's exists=True guard should reject before we touch the FS."""
    result = runner.invoke(app, ["index", "/this/path/definitely/does/not/exist"])
    assert result.exit_code != 0


def test_index_empty_directory_warns_and_exits_nonzero(tmp_path):
    """An empty dir is a user mistake — we don't silently 'succeed'."""
    result = runner.invoke(app, ["index", str(tmp_path), "--kb", "empty"])
    assert result.exit_code == 1


# ----- behavioral wiring (commands invoke the right core fns) ------------


def test_ask_invokes_retrieve_and_generate(fake_openai, fake_es, monkeypatch):
    """The `ask` command wires retrieve → generate → render. Mock both and
    verify the question and kb are threaded through."""
    from ragkit.core.retriever import RetrievedChunk
    captured = {}

    def fake_retrieve(question, kb_name, **kwargs):
        captured["question"] = question
        captured["kb"] = kb_name
        return [
            RetrievedChunk(
                rank=1, document_id="d1", document_name="doc.pdf",
                content="alpha content", similarity=0.9,
                vector_similarity=0.9, term_similarity=0.9,
            )
        ]

    # commands.py imports `retrieve` lazily inside cmd_ask, so the rebound
    # attribute on the source module is what matters.
    monkeypatch.setattr("ragkit.core.retriever.retrieve", fake_retrieve)

    fake_openai.chat_script = [("content", "Beta answer.")]

    result = runner.invoke(app, ["ask", "What is alpha?", "--kb", "finance"])

    assert result.exit_code == 0
    assert captured["question"] == "What is alpha?"
    assert captured["kb"] == "finance"
    assert "Beta answer." in result.stdout


def test_kb_list_outputs_known_names(fake_es):
    fake_es.list_indices.return_value = ["alpha", "beta"]
    result = runner.invoke(app, ["kb", "list"])
    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert "beta" in result.stdout


def test_kb_delete_aborts_without_confirmation(fake_es):
    """Destructive op — confirmation prompt must default to NO."""
    result = runner.invoke(app, ["kb", "delete", "finance"], input="n\n")
    assert result.exit_code == 0
    fake_es.delete_index.assert_not_called()


def test_kb_delete_with_yes_flag_proceeds(fake_es):
    """--yes flag exists so scripts can automate cleanup."""
    fake_es.delete_index.return_value = True
    result = runner.invoke(app, ["kb", "delete", "finance", "--yes"])
    assert result.exit_code == 0
    fake_es.delete_index.assert_called_once_with("finance")


def test_doctor_exits_nonzero_when_api_key_missing(monkeypatch, fake_es):
    """`rag doctor` is the first thing users run — must catch a missing key."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    fake_es.es.ping.return_value = True

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code != 0
    assert "DASHSCOPE_API_KEY" in result.stdout


def test_doctor_passes_with_full_setup(monkeypatch, fake_es):
    """Healthy setup → exit 0, with all checks green."""
    fake_es.es.ping.return_value = True
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Elasticsearch" in result.stdout
