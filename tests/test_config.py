"""Config loading — env-var precedence, missing-key contract."""

from __future__ import annotations

import pytest

from ragkit.config import Config, get_config


def test_config_reads_environment(monkeypatch):
    """Config picks up env vars set after import (re-read each call)."""
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen-max")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "768")
    monkeypatch.setenv("ES_HOST", "http://es.example:9200")

    cfg = get_config()

    assert cfg.llm_model == "qwen-max"
    assert cfg.embedding_dim == 768
    assert cfg.es_host == "http://es.example:9200"


def test_config_defaults_when_env_missing(monkeypatch):
    """Missing-but-non-required vars fall back to sensible defaults."""
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_DIM", raising=False)

    cfg = get_config()

    assert cfg.llm_model == "qwen-plus"
    assert cfg.embedding_model == "text-embedding-v3"
    assert cfg.embedding_dim == 1024


def test_config_is_frozen(monkeypatch):
    """Config is immutable — protects against accidental mutation."""
    cfg = get_config()
    with pytest.raises((AttributeError, Exception)):
        cfg.llm_model = "tampered"  # type: ignore[misc]


def test_require_api_key_raises_when_unset(monkeypatch):
    """The CLI relies on this guard to fail fast with a helpful message."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    cfg = get_config()
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        cfg.require_api_key()


def test_require_api_key_passes_when_set(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-real-looking-key")
    cfg = get_config()
    cfg.require_api_key()  # must not raise
