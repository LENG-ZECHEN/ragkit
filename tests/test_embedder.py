"""Embedder behavior — single, batch, and batch-splitting at the API limit."""

from __future__ import annotations

import pytest

from ragkit.core.embedder import _MAX_BATCH, embed_batch, embed_one


def test_embed_one_returns_single_vector(fake_openai):
    vec = embed_one("一些文本")
    assert isinstance(vec, list)
    assert len(vec) == 4  # matches conftest's RAG_EMBEDDING_DIM


def test_embed_batch_preserves_order(fake_openai):
    """Returned vectors must align 1:1 with input order — downstream
    indexing zips them with the original chunks."""
    texts = ["alpha", "beta", "gamma"]
    vectors = embed_batch(texts)

    assert len(vectors) == 3
    # Each text must produce its OWN vector — guard against off-by-one drift.
    assert vectors[0] != vectors[1]
    assert vectors[1] != vectors[2]


def test_embed_batch_splits_at_dashscope_limit(fake_openai):
    """DashScope caps batches at 10. Embedding 23 items should issue 3 calls,
    not 1 giant call, and not 23 single calls."""
    texts = [f"text-{i}" for i in range(23)]

    vectors = embed_batch(texts)

    assert len(vectors) == 23
    embed_calls = [c for c in fake_openai.calls if c["kind"] == "embed"]
    assert len(embed_calls) == 3
    assert [c["n"] for c in embed_calls] == [10, 10, 3]


def test_embed_batch_exactly_at_limit_issues_one_call(fake_openai):
    """Off-by-one regression guard: 10 items → one call, not two."""
    texts = [f"t{i}" for i in range(_MAX_BATCH)]
    embed_batch(texts)
    embed_calls = [c for c in fake_openai.calls if c["kind"] == "embed"]
    assert len(embed_calls) == 1


def test_embed_batch_empty_input_no_call(fake_openai):
    """Empty input should not call the API at all (waste of credits)."""
    vectors = embed_batch([])
    assert vectors == []
    assert not fake_openai.calls
