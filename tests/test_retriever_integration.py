"""Integration test: real index -> retrieve round-trip against Elasticsearch.

This is the one test that exercises the *real* hybrid path (BM25 + dense + rerank)
end-to-end, which every unit test mocks out. It is skipped unless BOTH of these
are set, so a plain ``pytest`` run skips it cleanly:

    RAGKIT_IT_ES_HOST            e.g. http://localhost:9200
    RAGKIT_IT_DASHSCOPE_API_KEY  a real DashScope key (used for embed + rerank)

Run it against a live ES + real key:

    RAGKIT_IT_ES_HOST=http://localhost:9200 \\
    RAGKIT_IT_DASHSCOPE_API_KEY=sk-... \\
    pytest -m integration tests/test_retriever_integration.py

Optional overrides: RAGKIT_IT_ES_USER, RAGKIT_IT_ES_PASSWORD (default to the
project's docker-compose credentials).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ES_HOST = os.environ.get("RAGKIT_IT_ES_HOST")
_API_KEY = os.environ.get("RAGKIT_IT_DASHSCOPE_API_KEY")


@pytest.fixture
def _real_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point config at real ES + DashScope, undoing conftest's mock env.

    conftest's autouse ``isolated_env`` fixture sets fast fakes (es.invalid,
    a fake key, 4-dim embeddings) for unit tests; restore real values here.
    """
    if not (_ES_HOST and _API_KEY):
        pytest.skip(
            "set RAGKIT_IT_ES_HOST and RAGKIT_IT_DASHSCOPE_API_KEY to run the "
            "Elasticsearch integration test"
        )
    monkeypatch.setenv("ES_HOST", _ES_HOST)
    monkeypatch.setenv("DASHSCOPE_API_KEY", _API_KEY)
    monkeypatch.setenv("ES_USER", os.environ.get("RAGKIT_IT_ES_USER", "elastic"))
    monkeypatch.setenv(
        "ES_PASSWORD", os.environ.get("RAGKIT_IT_ES_PASSWORD", "infini_rag_flow")
    )
    # Drop conftest's fast fakes so Config.from_env() falls back to the real
    # defaults (real embedding model + 1024 dims + real DashScope base URL).
    for fake in (
        "DASHSCOPE_BASE_URL",
        "RAG_EMBEDDING_MODEL",
        "RAG_EMBEDDING_DIM",
        "RAG_LLM_MODEL",
    ):
        monkeypatch.delenv(fake, raising=False)


def test_index_then_retrieve_ranks_relevant_doc(
    _real_services: None, tmp_path: Path
) -> None:
    from ragkit.core.indexer import index_file
    from ragkit.core.kb_manager import delete_kb
    from ragkit.core.retriever import retrieve

    kb = f"ragkit_it_{uuid.uuid4().hex[:8]}"
    docs = {
        "cats.txt": "Cats are small domesticated felines that purr and chase mice.",
        "finance.txt": "Quarterly revenue grew on strong cloud and advertising sales.",
        "weather.txt": "A cold front brings heavy rain and gusty winds tomorrow.",
    }
    try:
        for name, body in docs.items():
            path = tmp_path / name
            path.write_text(body, encoding="utf-8")
            index_file(path, kb_name=kb)

        chunks = retrieve("What drove the revenue growth?", kb_name=kb, top_k=3)

        # The relevant document must surface in the top-k.
        names = {c.document_name for c in chunks}
        assert "finance.txt" in names, f"relevant doc missing from top-k: {names}"

        # Scores must be meaningful: not all zero, and not all identical
        # (a degenerate ranking would be a real regression).
        scores = [c.similarity for c in chunks]
        assert any(s > 0 for s in scores), f"all similarities non-positive: {scores}"
        assert len(set(scores)) > 1, f"all similarities identical: {scores}"
    finally:
        delete_kb(kb)
