"""Shared pytest fixtures.

The tests are designed to run **without external services**:
- DashScope is replaced by an in-memory fake OpenAI client
- Elasticsearch is replaced by a fake ESConnection

This lets us run the full RAG pipeline behaviorally in CI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean, known environment.

    Critically: set DASHSCOPE_API_KEY so config.require_api_key() doesn't fail
    in tests that don't care about it.
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-not-real")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen-plus")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "text-embedding-v3")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "4")  # small dim for fast tests
    monkeypatch.setenv("ES_HOST", "http://es.invalid:9200")
    monkeypatch.setenv("ES_USER", "elastic")
    monkeypatch.setenv("ES_PASSWORD", "test")


# ---------------------------------------------------------------------------
# Fake OpenAI client (covers embedding + chat completion)
# ---------------------------------------------------------------------------


class _FakeEmbeddingData:
    def __init__(self, vector: list[float]):
        self.embedding = vector


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]):
        self.data = [_FakeEmbeddingData(v) for v in vectors]


class _FakeChatDelta:
    def __init__(self, content: str | None = None, reasoning: str | None = None):
        self.content = content
        self.reasoning_content = reasoning


class _FakeChatChoice:
    def __init__(
        self,
        content: str | None = None,
        reasoning: str | None = None,
        finish_reason: str | None = None,
    ):
        self.delta = _FakeChatDelta(content=content, reasoning=reasoning)
        self.finish_reason = finish_reason


class _FakeChatChunk:
    def __init__(self, choices: list[_FakeChatChoice]):
        self.choices = choices


class FakeOpenAI:
    """A tiny stand-in for openai.OpenAI used in tests.

    - embeddings.create returns deterministic vectors derived from input text.
    - chat.completions.create returns a scripted token stream.
    """

    def __init__(
        self,
        *,
        chat_script: list[tuple[str, str]] | None = None,
        embedding_dim: int = 4,
    ):
        self.embedding_dim = embedding_dim
        self.chat_script = chat_script or [("content", "Hello world.")]
        self.calls: list[dict[str, Any]] = []
        self.embeddings = _Embeddings(self)
        self.chat = _Chat(self)


class _Embeddings:
    def __init__(self, parent: FakeOpenAI):
        self._parent = parent

    def create(self, *, model: str, input: Any, dimensions: int, encoding_format: str):
        # Accept str or list[str]; normalize to list.
        items = [input] if isinstance(input, str) else list(input)
        self._parent.calls.append({"kind": "embed", "n": len(items), "model": model})
        # Deterministic per-text vector so we can assert ordering later.
        vectors = [
            [float(hash(text + str(i)) % 1000) / 1000.0 for i in range(dimensions)]
            for text in items
        ]
        return _FakeEmbeddingResponse(vectors)


class _Chat:
    def __init__(self, parent: FakeOpenAI):
        self._parent = parent
        self.completions = _Completions(parent)


class _Completions:
    def __init__(self, parent: FakeOpenAI):
        self._parent = parent

    def create(self, *, model: str, messages: list[dict], stream: bool, extra_body=None):
        self._parent.calls.append({"kind": "chat", "model": model, "stream": stream})

        def gen() -> Iterator[_FakeChatChunk]:
            for kind, text in self._parent.chat_script:
                if kind == "content":
                    yield _FakeChatChunk([_FakeChatChoice(content=text)])
                elif kind == "thinking":
                    yield _FakeChatChunk([_FakeChatChoice(reasoning=text)])
            yield _FakeChatChunk([_FakeChatChoice(finish_reason="stop")])

        return gen()


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    """Patch openai.OpenAI everywhere we use it."""
    instance = FakeOpenAI()
    monkeypatch.setattr("ragkit.core.embedder.OpenAI", lambda **kw: instance)
    monkeypatch.setattr("ragkit.core.generator.OpenAI", lambda **kw: instance)
    return instance


# ---------------------------------------------------------------------------
# Fake Elasticsearch connection
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_es(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ESConnection() with a configurable mock.

    We patch the singleton-returning name in es_conn so any module that does
    `from ragkit.core.rag.utils.es_conn import ESConnection` and then
    `ESConnection()` gets the mock factory.
    """
    fake = MagicMock(name="ESConnection")
    # Default behaviors:
    fake.es.indices.exists.return_value = False
    fake.es.indices.get.return_value = {}
    fake.es.bulk.return_value = {"errors": False, "items": []}
    fake.insert.return_value = []
    fake.ensure_index.return_value = None
    fake.delete_index.return_value = True
    fake.list_indices.return_value = []

    # Patch the factory in every module that imports it lazily.
    monkeypatch.setattr(
        "ragkit.core.rag.utils.es_conn.ESConnection",
        lambda: fake,
    )
    monkeypatch.setattr("ragkit.core.indexer.ESConnection", lambda: fake)
    monkeypatch.setattr("ragkit.core.kb_manager.ESConnection", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """Write a small text file we can index end-to-end."""
    path = tmp_path / "sample.txt"
    path.write_text(
        "人工智能是一门新兴技术。\n"
        "在金融、医疗、教育等领域都有广泛应用。\n"
        "RAG 是 Retrieval-Augmented Generation 的缩写。\n",
        encoding="utf-8",
    )
    return path
