"""Tests for searcher.py — thin wrappers around ES queries for graph artifacts."""

from __future__ import annotations

import pytest

from ragkit.core.graph.searcher import (
    fetch_chunks_by_ids,
    search_communities_by_entity_names,
    search_communities_by_vector,
    search_entities_by_vector,
)


# ----- search_entities_by_vector ------------------------------------------


def test_search_entities_returns_empty_when_index_absent(fake_openai, fake_es):
    """Index doesn't exist yet — return [], don't crash."""
    fake_es.es.indices.exists.return_value = False
    assert search_entities_by_vector("kb", "any question") == []


def test_search_entities_filters_by_type_kwd(fake_openai, fake_es):
    """The kNN query must filter type_kwd=entity."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {
        "hits": {"hits": [{"_source": {"entity_name_kwd": "qwen"}}]}
    }

    search_entities_by_vector("kb", "q", top_k=5)

    call = fake_es.es.search.call_args
    knn = call.kwargs.get("knn")
    assert knn is not None
    # The kNN filter must restrict to entity docs.
    assert knn["filter"]["term"]["type_kwd"] == "entity"
    assert knn["k"] == 5


def test_search_entities_extracts_source(fake_openai, fake_es):
    """The wrapper must unpack _source from ES hits."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {
        "hits": {"hits": [
            {"_source": {"entity_name_kwd": "qwen"}},
            {"_source": {"entity_name_kwd": "dashscope"}},
        ]}
    }

    out = search_entities_by_vector("kb", "q")
    assert [e["entity_name_kwd"] for e in out] == ["qwen", "dashscope"]


def test_search_entities_handles_query_failure(fake_openai, fake_es):
    """ES error must not propagate — return []."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.side_effect = RuntimeError("ES down")
    assert search_entities_by_vector("kb", "q") == []


# ----- search_communities_by_vector ---------------------------------------


def test_search_communities_default_no_level_filter(fake_openai, fake_es):
    """No level → only type filter, no level filter."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {"hits": {"hits": []}}

    search_communities_by_vector("kb", "q")

    knn = fake_es.es.search.call_args.kwargs["knn"]
    must = knn["filter"]["bool"]["must"]
    types = [m["term"].get("type_kwd") for m in must if "type_kwd" in m["term"]]
    levels = [m["term"].get("community_level_int") for m in must if "community_level_int" in m["term"]]
    assert types == ["community"]
    assert levels == []  # no level filter


def test_search_communities_applies_level_filter(fake_openai, fake_es):
    """level=2 → must include community_level_int=2 filter."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {"hits": {"hits": []}}

    search_communities_by_vector("kb", "q", level=2)

    knn = fake_es.es.search.call_args.kwargs["knn"]
    must = knn["filter"]["bool"]["must"]
    levels = [m["term"].get("community_level_int") for m in must if "community_level_int" in m["term"]]
    assert levels == [2]


# ----- search_communities_by_entity_names ---------------------------------


def test_search_communities_by_names_empty_input_returns_empty(fake_es):
    """Empty seed list → skip ES query entirely."""
    assert search_communities_by_entity_names("kb", []) == []
    fake_es.es.search.assert_not_called()


def test_search_communities_by_names_uses_terms_query(fake_openai, fake_es):
    """Must use terms filter on community_entity_names_kwd."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.search.return_value = {"hits": {"hits": []}}

    search_communities_by_entity_names("kb", ["qwen", "dashscope"], top_k=5)

    body = fake_es.es.search.call_args.kwargs["query"]
    should = body["bool"]["should"]
    terms_clauses = [s["terms"]["community_entity_names_kwd"] for s in should if "terms" in s]
    assert ["qwen", "dashscope"] in terms_clauses


def test_search_communities_by_names_handles_missing_index(fake_es):
    fake_es.es.indices.exists.return_value = False
    assert search_communities_by_entity_names("kb", ["x"]) == []


# ----- fetch_chunks_by_ids ------------------------------------------------


def test_fetch_chunks_returns_found_only(fake_es):
    """Only `found=True` docs are returned; missing IDs are silently dropped."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.mget.return_value = {
        "docs": [
            {"_id": "c1", "found": True, "_source": {"content_with_weight": "T1"}},
            {"_id": "c2", "found": False},   # missing — drop
            {"_id": "c3", "found": True, "_source": {"content_with_weight": "T3"}},
        ]
    }

    out = fetch_chunks_by_ids("kb", ["c1", "c2", "c3"])
    ids = {c["_id"] for c in out}
    assert ids == {"c1", "c3"}


def test_fetch_chunks_deduplicates_input_ids(fake_es):
    """Even with duplicate IDs, only one mget call is made."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.mget.return_value = {"docs": []}

    fetch_chunks_by_ids("kb", ["c1", "c1", "c2", "c2"])

    call = fake_es.es.mget.call_args
    requested_ids = call.kwargs.get("ids") or call.args[1] if call.args else call.kwargs["ids"]
    assert sorted(requested_ids) == ["c1", "c2"]


def test_fetch_chunks_empty_input_skips_es_call(fake_es):
    assert fetch_chunks_by_ids("kb", []) == []
    fake_es.es.mget.assert_not_called()


def test_fetch_chunks_handles_missing_index(fake_es):
    fake_es.es.indices.exists.return_value = False
    assert fetch_chunks_by_ids("kb", ["c1"]) == []
    fake_es.es.mget.assert_not_called()


def test_fetch_chunks_handles_es_error(fake_es):
    """ES error must not propagate."""
    fake_es.es.indices.exists.return_value = True
    fake_es.es.mget.side_effect = RuntimeError("ES down")
    assert fetch_chunks_by_ids("kb", ["c1"]) == []
