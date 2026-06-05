"""Tests for the structured CommunityReport generation pipeline.

Covers _parse_report's robustness (code fences, malformed JSON, missing
fields, oversized findings) and the in-place mutation contract of
generate_community_report / summarize_all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.core.graph.store import NetworkXGraphStore
from ragkit.core.graph.summarizer import (
    MAX_FINDINGS_PER_COMMUNITY,
    _parse_report,
    _strip_code_fence,
    generate_community_report,
    summarize_all,
)
from ragkit.core.graph.types import Community, Entity, Finding, Relation


def _store(tmp_path: Path) -> NetworkXGraphStore:
    s = NetworkXGraphStore(path=tmp_path / "g.json")
    s.upsert_entity(Entity(name="qwen", type="model", description="阿里大模型"))
    s.upsert_entity(Entity(name="dashscope", type="platform", description="阿里平台"))
    s.upsert_relation(Relation(source="qwen", target="dashscope", description="部署在"))
    s.set_communities([Community(id=0, entity_names=["qwen", "dashscope"])])
    return s


# ----- _strip_code_fence --------------------------------------------------


def test_strip_code_fence_handles_json_fence():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_passes_through_plain():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


# ----- _parse_report ------------------------------------------------------


def _community() -> Community:
    return Community(id=42, entity_names=["x", "y"])


def test_parse_report_extracts_all_fields():
    raw = json.dumps({
        "title": "国产大模型生态",
        "summary": "这个群组讨论...",
        "rank": 8,
        "rank_explanation": "重要因为...",
        "findings": [
            {"summary": "f1", "explanation": "e1"},
            {"summary": "f2", "explanation": "e2"},
        ],
    })
    out = _parse_report(raw, _community())
    assert out["title"] == "国产大模型生态"
    assert out["summary"] == "这个群组讨论..."
    assert out["rank"] == 8.0
    assert out["rank_explanation"] == "重要因为..."
    assert len(out["findings"]) == 2
    assert out["findings"][0]["summary"] == "f1"


def test_parse_report_handles_code_fence_wrapping():
    raw = '```json\n' + json.dumps({"title": "T", "summary": "S"}) + '\n```'
    out = _parse_report(raw, _community())
    assert out["title"] == "T"
    assert out["summary"] == "S"


def test_parse_report_caps_findings_at_max():
    raw = json.dumps({
        "title": "T",
        "summary": "S",
        "rank": 5,
        "findings": [{"summary": f"f{i}", "explanation": "e"} for i in range(20)],
    })
    out = _parse_report(raw, _community())
    assert len(out["findings"]) == MAX_FINDINGS_PER_COMMUNITY


def test_parse_report_drops_blank_findings():
    raw = json.dumps({
        "title": "T",
        "summary": "S",
        "findings": [
            {"summary": "good", "explanation": "yes"},
            {"summary": "", "explanation": ""},  # blank — drop
            {"summary": "good2", "explanation": ""},  # one half is fine
        ],
    })
    out = _parse_report(raw, _community())
    assert len(out["findings"]) == 2  # blank one dropped


def test_parse_report_clamps_rank_to_range():
    """LLMs sometimes return out-of-range scores; clamp to [0, 10]."""
    raw = json.dumps({"title": "T", "summary": "S", "rank": 99, "findings": []})
    out = _parse_report(raw, _community())
    assert out["rank"] == 10.0

    raw = json.dumps({"title": "T", "summary": "S", "rank": -5, "findings": []})
    out = _parse_report(raw, _community())
    assert out["rank"] == 0.0


def test_parse_report_handles_non_numeric_rank():
    """A string rank like 'high' must not crash."""
    raw = json.dumps({"title": "T", "summary": "S", "rank": "high", "findings": []})
    out = _parse_report(raw, _community())
    assert out["rank"] == 0.0


def test_parse_report_fills_defaults_on_missing_fields():
    raw = json.dumps({"title": "T"})  # everything else missing
    out = _parse_report(raw, _community())
    assert out["title"] == "T"
    assert out["summary"] == ""
    assert out["rank"] == 0.0
    assert out["findings"] == []


def test_parse_report_returns_safe_default_on_bad_json():
    """Malformed JSON → return a minimal dict, don't crash."""
    out = _parse_report("not even json", _community())
    assert out["title"] == "Community 42"
    assert out["summary"] == ""
    assert out["findings"] == []


# ----- generate_community_report (end-to-end) -----------------------------


def test_generate_community_report_fills_all_fields(tmp_path, fake_openai):
    """Happy path: LLM returns full JSON, all Community fields get populated."""
    fake_openai.chat_script = [("content", json.dumps({
        "title": "Qwen 生态",
        "summary": "通义千问及其 DashScope 部署平台",
        "rank": 7,
        "rank_explanation": "核心产品",
        "findings": [
            {"summary": "qwen 是阿里旗舰", "explanation": "..."},
        ],
    }))]
    store = _store(tmp_path)
    community = store.all_communities()[0]

    generate_community_report(community, store)

    assert community.title == "Qwen 生态"
    assert "DashScope" in community.summary
    assert community.rank == 7.0
    assert community.rank_explanation == "核心产品"
    assert len(community.findings) == 1
    assert isinstance(community.findings[0], Finding)
    assert community.findings[0].summary == "qwen 是阿里旗舰"


def test_generate_community_report_empty_community_noop(tmp_path, fake_openai):
    """Empty entity list → no LLM call, fields remain default."""
    store = _store(tmp_path)
    community = Community(id=99, entity_names=[])

    generate_community_report(community, store)

    assert community.title == ""
    assert community.findings == []
    # No chat call should have been made.
    chat_calls = [c for c in fake_openai.calls if c["kind"] == "chat"]
    assert len(chat_calls) == 0


def test_generate_community_report_llm_failure_keeps_defaults(
    tmp_path, fake_openai, monkeypatch
):
    """LLM error: community fields stay at their pre-call values."""
    def boom(**kw):
        raise RuntimeError("API down")

    monkeypatch.setattr(fake_openai.chat.completions, "create", boom)

    store = _store(tmp_path)
    community = store.all_communities()[0]
    original_title = community.title  # ""

    generate_community_report(community, store)

    assert community.title == original_title
    assert community.findings == []


# ----- summarize_all batch contract --------------------------------------


def test_summarize_all_calls_llm_per_community(tmp_path, fake_openai):
    fake_openai.chat_script = [("content", json.dumps({
        "title": "T", "summary": "S", "rank": 5, "findings": [],
    }))]
    store = _store(tmp_path)
    store.set_communities([
        Community(id=0, entity_names=["qwen", "dashscope"]),
        Community(id=1, entity_names=["qwen"]),
    ])

    failures = summarize_all(store)

    assert failures == 0
    chat_calls = [c for c in fake_openai.calls if c["kind"] == "chat"]
    assert len(chat_calls) == 2


def test_summarize_all_respects_max_communities_without_dropping(
    tmp_path, fake_openai
):
    """Regression test for the data-loss bug fixed earlier: max_communities=1
    means only the first gets a report, but the rest stay in the store."""
    fake_openai.chat_script = [("content", json.dumps({
        "title": "T", "summary": "S", "rank": 5, "findings": [],
    }))]
    store = _store(tmp_path)
    store.set_communities([
        Community(id=0, entity_names=["qwen", "dashscope"]),
        Community(id=1, entity_names=["qwen"]),
        Community(id=2, entity_names=["dashscope"]),
    ])

    summarize_all(store, max_communities=1)

    saved = store.all_communities()
    assert len(saved) == 3  # ALL kept
    assert saved[0].title == "T"  # First one got summarized


def test_summarize_all_fires_progress_callback(tmp_path, fake_openai):
    fake_openai.chat_script = [("content", json.dumps({
        "title": "T", "summary": "S", "rank": 5, "findings": [],
    }))]
    store = _store(tmp_path)
    store.set_communities([
        Community(id=0, entity_names=["qwen", "dashscope"]),
        Community(id=1, entity_names=["qwen"]),
    ])

    stages: list[tuple[str, int, int]] = []

    def cb(stage, current, total):
        stages.append((stage, current, total))

    summarize_all(store, progress_cb=cb)

    assert len(stages) == 2
    assert all(s[0] == "summarizing" for s in stages)
    # current should go 1, 2 ...
    assert stages[0][1] == 1
    assert stages[1][1] == 2


# ----- ISS-013: edge cases for summarize_all ------------------------------


def test_summarize_all_with_empty_store_makes_no_llm_calls(tmp_path, fake_openai):
    """Empty store → no chat calls, no failures, early-return path."""
    store = NetworkXGraphStore(path=tmp_path / "g.json")
    # No communities set at all.
    failures = summarize_all(store)
    assert failures == 0
    chat_calls = [c for c in fake_openai.calls if c["kind"] == "chat"]
    assert chat_calls == []


def test_summarize_all_with_zero_cap_processes_nothing(tmp_path, fake_openai):
    """max_communities=0 must mean 'process none', NOT 'process all'.

    Off-by-one regression here would be hard to spot otherwise: 0 could
    accidentally bypass the slice and process everything if interpreted
    as falsy → None.
    """
    fake_openai.chat_script = [("content", json.dumps({
        "title": "T", "summary": "S", "rank": 5, "findings": [],
    }))]
    store = _store(tmp_path)
    store.set_communities([
        Community(id=0, entity_names=["qwen", "dashscope"]),
        Community(id=1, entity_names=["qwen"]),
    ])

    failures = summarize_all(store, max_communities=0)
    assert failures == 0
    chat_calls = [c for c in fake_openai.calls if c["kind"] == "chat"]
    assert chat_calls == []
    # Communities themselves preserved.
    assert len(store.all_communities()) == 2
