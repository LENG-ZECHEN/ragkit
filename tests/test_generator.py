"""Generator behavior — prompt building, event-stream parsing."""

from __future__ import annotations

import pytest

from ragkit.core.generator import Event, build_prompt, generate
from ragkit.core.retriever import RetrievedChunk


def _chunk(rank: int, content: str, name: str = "doc.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        document_id=f"d{rank}",
        document_name=name,
        content=content,
        similarity=0.9,
        vector_similarity=0.9,
        term_similarity=0.9,
    )


# ----- prompt building ----------------------------------------------------


def test_prompt_inserts_chunks_with_numeric_tags():
    """Citation tags ##N$$ depend on chunks being numbered [1], [2], ... ."""
    chunks = [_chunk(1, "Alpha facts."), _chunk(2, "Beta facts.")]
    prompt = build_prompt("What is alpha?", chunks)

    assert "[1] Alpha facts." in prompt
    assert "[2] Beta facts." in prompt
    assert "What is alpha?" in prompt


def test_prompt_handles_empty_references():
    """No chunks → prompt must still be valid (model decides whether to refuse)."""
    prompt = build_prompt("Hello", [])
    assert "Hello" in prompt
    assert "暂无相关参考内容" in prompt


def test_prompt_includes_citation_instruction():
    """The ##N$$ format is load-bearing for UI rendering — guard it."""
    prompt = build_prompt("q", [_chunk(1, "c")])
    assert "##" in prompt and "$$" in prompt


# ----- streaming event flow ----------------------------------------------


def test_generate_emits_content_then_done(fake_openai):
    """The CLI relies on every successful run ending with type='done'."""
    fake_openai.chat_script = [
        ("content", "Hello "),
        ("content", "world."),
    ]
    chunks = [_chunk(1, "context")]

    events = list(generate("q", chunks))

    types = [e.type for e in events]
    assert types == ["content", "content", "done"]
    assert "".join(e.text for e in events if e.type == "content") == "Hello world."


def test_generate_separates_thinking_from_content(fake_openai):
    """Reasoning tokens go on a different channel — REPL hides them by default."""
    fake_openai.chat_script = [
        ("thinking", "Let me think... "),
        ("content", "The answer is 42."),
    ]
    events = list(generate("q", []))

    assert any(e.type == "thinking" for e in events)
    assert any(e.type == "content" for e in events)
    answer = "".join(e.text for e in events if e.type == "content")
    assert answer == "The answer is 42."


def test_generate_attaches_references_to_done(fake_openai):
    """The terminal 'done' event must carry the chunks so the UI can render
    the citations table after the answer."""
    fake_openai.chat_script = [("content", "ok")]
    chunks = [_chunk(1, "a"), _chunk(2, "b")]

    events = list(generate("q", chunks))

    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert len(done[0].references) == 2
    assert done[0].references[0].rank == 1


def test_generate_yields_error_event_on_exception(fake_openai, monkeypatch):
    """LLM failures must NOT raise — they become error events the CLI can show."""
    def boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(fake_openai.chat.completions, "create", boom)

    events = list(generate("q", []))

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "network down" in error_events[0].text
