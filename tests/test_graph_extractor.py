"""Extractor — JSON parsing robustness & validation of LLM output."""

from __future__ import annotations

import json

import pytest

from ragkit.core.graph.extractor import (
    ExtractionResult,
    _parse_extraction,
    _strip_code_fence,
    extract_from_text,
)


# ----- LLM output cleaning -----------------------------------------------


def test_strip_code_fence_handles_json_fence():
    raw = "```json\n{\"a\": 1}\n```"
    assert _strip_code_fence(raw) == '{"a": 1}'


def test_strip_code_fence_handles_plain_fence():
    assert _strip_code_fence("```\n{\"a\": 1}\n```") == '{"a": 1}'


def test_strip_code_fence_passes_through_plain_json():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


# ----- parse_extraction --------------------------------------------------


def test_parse_extracts_entities_and_relations():
    raw = json.dumps({
        "entities": [
            {"name": "OpenAI", "type": "organization", "description": "AI lab"},
            {"name": "GPT-4", "type": "concept", "description": "LLM by OpenAI"},
        ],
        "relations": [
            {"source": "OpenAI", "target": "GPT-4", "description": "creates"},
        ],
    })
    result = _parse_extraction(raw, chunk_id="c1")

    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert all(e.source_chunks == ["c1"] for e in result.entities)
    assert result.relations[0].source.lower() == "openai"


def test_parse_drops_relations_with_dangling_endpoints():
    """LLM sometimes references entities it didn't put in the entity list.
    Those edges would point to nothing — must be dropped, not silently kept."""
    raw = json.dumps({
        "entities": [{"name": "X", "type": "t", "description": ""}],
        "relations": [
            {"source": "X", "target": "Y_NOT_DEFINED", "description": "?"},
            {"source": "Z_NOT_DEFINED", "target": "X", "description": "?"},
        ],
    })
    result = _parse_extraction(raw, chunk_id="c1")

    assert len(result.entities) == 1
    assert result.relations == []  # Both bad edges dropped


def test_parse_drops_self_loops_at_extraction():
    raw = json.dumps({
        "entities": [{"name": "X", "type": "t", "description": ""}],
        "relations": [{"source": "X", "target": "X", "description": "self"}],
    })
    result = _parse_extraction(raw, chunk_id="c1")
    assert result.relations == []


def test_parse_dedupes_entities_case_insensitively():
    """LLMs sometimes emit 'OpenAI' and 'openai' as separate entities — merge."""
    raw = json.dumps({
        "entities": [
            {"name": "OpenAI", "type": "org", "description": ""},
            {"name": "openai", "type": "org", "description": ""},
        ],
        "relations": [],
    })
    result = _parse_extraction(raw, chunk_id="c1")
    assert len(result.entities) == 1


def test_parse_returns_empty_on_bad_json():
    """Malformed JSON shouldn't crash the indexer — just log and yield nothing."""
    result = _parse_extraction("not even close to json", chunk_id="c1")
    assert result == ExtractionResult(entities=[], relations=[])


def test_parse_handles_empty_arrays():
    """LLM correctly returned 'no entities' — must not crash."""
    raw = json.dumps({"entities": [], "relations": []})
    result = _parse_extraction(raw, chunk_id="c1")
    assert result.entities == []
    assert result.relations == []


def test_parse_drops_blank_names():
    """An entity with empty name is junk — drop it instead of polluting the graph."""
    raw = json.dumps({
        "entities": [
            {"name": "", "type": "t", "description": "blank"},
            {"name": "real", "type": "t", "description": "ok"},
        ],
        "relations": [],
    })
    result = _parse_extraction(raw, chunk_id="c1")
    assert len(result.entities) == 1
    assert result.entities[0].name == "real"


# ----- end-to-end extract_from_text --------------------------------------


def test_extract_from_text_empty_returns_empty(fake_openai):
    """Don't burn an LLM call on empty input."""
    result = extract_from_text("", chunk_id="c1")
    assert result.entities == [] and result.relations == []
    chat_calls = [c for c in fake_openai.calls if c["kind"] == "chat"]
    assert len(chat_calls) == 0


def test_extract_from_text_returns_empty_on_llm_failure(fake_openai, monkeypatch):
    """LLM errors must NOT propagate — graph build should keep going on
    other chunks even if one extraction fails."""
    def boom(**kwargs):
        raise RuntimeError("API down")

    monkeypatch.setattr(fake_openai.chat.completions, "create", boom)

    result = extract_from_text("some text", chunk_id="c1")
    assert result.entities == []
    assert result.relations == []


def test_extract_from_text_parses_scripted_response(fake_openai):
    """Full path: text → LLM → parse → Entity list."""
    payload = json.dumps({
        "entities": [{"name": "Alpha", "type": "concept", "description": "the first letter"}],
        "relations": [],
    })
    fake_openai.chat_script = [("content", payload)]
    result = extract_from_text("Alpha is the first Greek letter.", chunk_id="ch1")
    assert len(result.entities) == 1
    assert result.entities[0].name == "Alpha"
