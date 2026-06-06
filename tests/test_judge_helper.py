"""Tests for ``evals.judge_helper`` (P2.5 human-in-the-loop judging)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from evals.judge_helper import (
    _MERGED_COLUMNS,
    _cli_main,
    _next_batch_number,
    _params_key,
    _row_key,
    merge_to_csv,
    read_unjudged,
    write_judges,
)
from evals.schema import QAItem, save_dataset


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


def _trace_row(
    *, qa_id: str, mode: str, params: dict[str, Any],
    answer: str = "ans", chunks: list[str] | None = None,
    recall_at_k: dict[str, float] | None = None,
    mrr: float = 0.5, ndcg: float = 0.5, refusal_correct: bool = False,
) -> dict[str, Any]:
    """Build a SweepResultRow-shaped dict for traces.jsonl."""
    chunks = chunks if chunks is not None else ["c1"]
    return {
        "qa_id": qa_id,
        "mode": mode,
        "params": params,
        "trace": {
            "answer": answer,
            "timing": {
                "embed_ms": 1.0, "retrieve_es_ms": 2.0, "rerank_ms": 0.0,
                "generate_ms": 3.0, "total_ms": 6.0,
            },
            "cost": {
                "llm_calls": 1, "embedding_calls": 1,
                "tokens_in": 100, "tokens_out": 50, "est_cost_usd": 0.0001,
            },
        },
        "recall_at_k": recall_at_k or {"1": 1.0, "3": 1.0, "5": 1.0, "10": 1.0},
        "mrr": mrr,
        "ndcg_at_10": ndcg,
        "refusal_correct": refusal_correct,
        "retrieved_contents": [{"chunk_id": cid, "content": f"<{cid}>"}
                               for cid in chunks],
    }


def _write_traces(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _judge(qa_id: str, mode: str, params: dict[str, Any],
           *, f: int = 4, r: int = 4, c: int = 4) -> dict[str, Any]:
    return {
        "qa_id": qa_id, "mode": mode, "params": params,
        "faithfulness": f, "faithfulness_reason": "fr",
        "relevance": r, "relevance_reason": "rr",
        "completeness": c, "completeness_reason": "cr",
    }


def _ds(tmp_path: Path) -> Path:
    items = [
        QAItem(id="fact-001", question="Q1?", category="factual",
               ground_truth_chunk_ids=["c1"], gold_answer="A1"),
        QAItem(id="fact-002", question="Q2?", category="factual",
               ground_truth_chunk_ids=["c2"], gold_answer="A2"),
        QAItem(id="refuse-001", question="Q3?", category="refusal",
               ground_truth_chunk_ids=[], gold_answer=None),
    ]
    path = tmp_path / "dataset.jsonl"
    save_dataset(items, path)
    return path


# --------------------------------------------------------------------------
# _params_key / _row_key
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_params_key_order_insensitive():
    assert _params_key({"a": 1, "b": 2}) == _params_key({"b": 2, "a": 1})


@pytest.mark.unit
def test_row_key_combines_all_three():
    k1 = _row_key("fact-001", "vector", {"vsw": 0.5})
    k2 = _row_key("fact-001", "vector", {"vsw": 0.5})
    k3 = _row_key("fact-001", "local", {"vsw": 0.5})
    assert k1 == k2
    assert k1 != k3


# --------------------------------------------------------------------------
# _next_batch_number
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_next_batch_number_starts_at_one(tmp_path):
    assert _next_batch_number(tmp_path / "empty") == 1


@pytest.mark.unit
def test_next_batch_number_increments(tmp_path):
    (tmp_path / "batch_001.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "batch_005.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "noise.txt").write_text("", encoding="utf-8")
    assert _next_batch_number(tmp_path) == 6


# --------------------------------------------------------------------------
# read_unjudged
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_read_unjudged_returns_all_when_no_judges(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-002", mode="vector", params={"vsw": 0.5}),
    ])
    judge_dir = tmp_path / "judges"
    batch = read_unjudged(traces, judge_dir, batch_size=50)
    assert len(batch) == 2
    # Each entry has the keys the human judge needs.
    for entry in batch:
        for key in ("qa_id", "category", "question", "gold_answer",
                    "mode", "params", "generated_answer", "retrieved_contents"):
            assert key in entry


@pytest.mark.unit
def test_read_unjudged_skips_already_judged(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-002", mode="vector", params={"vsw": 0.5}),
    ])
    judge_dir = tmp_path / "judges"
    write_judges([_judge("fact-001", "vector", {"vsw": 0.5})], judge_dir)
    batch = read_unjudged(traces, judge_dir, batch_size=50)
    assert len(batch) == 1
    assert batch[0]["qa_id"] == "fact-002"


@pytest.mark.unit
def test_read_unjudged_respects_batch_size(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-002", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="refuse-001", mode="vector", params={"vsw": 0.5}),
    ])
    judge_dir = tmp_path / "judges"
    assert len(read_unjudged(traces, judge_dir, batch_size=2)) == 2
    assert len(read_unjudged(traces, judge_dir, batch_size=0)) == 0


@pytest.mark.unit
def test_read_unjudged_deterministic_order(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    # Insert in scrambled order.
    _write_traces(traces, [
        _trace_row(qa_id="fact-002", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-001", mode="local", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.2}),
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
    ])
    judge_dir = tmp_path / "judges"
    batch = read_unjudged(traces, judge_dir, batch_size=50)
    keys = [(r["mode"], r["qa_id"], _params_key(r["params"])) for r in batch]
    assert keys == sorted(keys)


@pytest.mark.unit
def test_read_unjudged_params_dict_order_irrelevant(tmp_path):
    """A judge written with keys in different order should still mark it judged."""
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector",
                   params={"vsw": 0.5, "st": 0.1}),
    ])
    judge_dir = tmp_path / "judges"
    write_judges([_judge("fact-001", "vector", {"st": 0.1, "vsw": 0.5})], judge_dir)
    assert read_unjudged(traces, judge_dir, batch_size=50) == []


@pytest.mark.unit
def test_read_unjudged_populates_dataset_context(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5},
                   answer="model_said"),
    ])
    batch = read_unjudged(traces, tmp_path / "judges", batch_size=10)
    row = batch[0]
    assert row["question"] == "Q1?"
    assert row["category"] == "factual"
    assert row["gold_answer"] == "A1"
    assert row["generated_answer"] == "model_said"
    assert row["retrieved_contents"] == [{"chunk_id": "c1", "content": "<c1>"}]


# --------------------------------------------------------------------------
# write_judges
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_write_judges_creates_batch_001(tmp_path):
    judge_dir = tmp_path / "judges"
    path = write_judges(
        [_judge("fact-001", "vector", {"vsw": 0.5})], judge_dir,
    )
    assert path.name == "batch_001.jsonl"
    assert path.exists()


@pytest.mark.unit
def test_write_judges_auto_increments(tmp_path):
    judge_dir = tmp_path / "judges"
    p1 = write_judges([_judge("a", "vector", {"v": 0.1})], judge_dir)
    p2 = write_judges([_judge("b", "vector", {"v": 0.1})], judge_dir)
    p3 = write_judges([_judge("c", "vector", {"v": 0.1})], judge_dir)
    assert (p1.name, p2.name, p3.name) == (
        "batch_001.jsonl", "batch_002.jsonl", "batch_003.jsonl",
    )


@pytest.mark.unit
def test_write_judges_rejects_empty_batch(tmp_path):
    with pytest.raises(ValueError):
        write_judges([], tmp_path / "judges")


@pytest.mark.unit
@pytest.mark.parametrize("bad_score", [0, 6, -1, 100])
def test_write_judges_rejects_out_of_range_score(tmp_path, bad_score):
    j = _judge("a", "vector", {"v": 0.1})
    j["faithfulness"] = bad_score
    with pytest.raises(ValueError):
        write_judges([j], tmp_path / "judges")


@pytest.mark.unit
def test_write_judges_rejects_non_int_score(tmp_path):
    j = _judge("a", "vector", {"v": 0.1})
    j["relevance"] = 3.5  # type: ignore[assignment]
    with pytest.raises(ValueError):
        write_judges([j], tmp_path / "judges")


@pytest.mark.unit
def test_write_judges_rejects_string_score(tmp_path):
    j = _judge("a", "vector", {"v": 0.1})
    j["completeness"] = "four"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        write_judges([j], tmp_path / "judges")


@pytest.mark.unit
def test_write_judges_rejects_missing_field(tmp_path):
    j = _judge("a", "vector", {"v": 0.1})
    del j["faithfulness_reason"]
    with pytest.raises(ValueError):
        write_judges([j], tmp_path / "judges")


@pytest.mark.unit
def test_write_judges_rejects_bool_score(tmp_path):
    # bool is a subclass of int in Python — guard against True passing through.
    j = _judge("a", "vector", {"v": 0.1})
    j["faithfulness"] = True  # type: ignore[assignment]
    with pytest.raises(ValueError):
        write_judges([j], tmp_path / "judges")


# --------------------------------------------------------------------------
# merge_to_csv
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_to_csv_column_order_and_count(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-002", mode="vector", params={"vsw": 0.5}),
    ])
    judge_dir = tmp_path / "judges"
    out = tmp_path / "metrics.csv"
    n = merge_to_csv(traces, judge_dir, out)
    assert n == 2

    with out.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    assert rows[0] == list(_MERGED_COLUMNS)
    assert len(rows) == 3  # header + 2 data rows


@pytest.mark.unit
def test_merge_to_csv_joins_judges_correctly(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-002", mode="vector", params={"vsw": 0.5}),
    ])
    judge_dir = tmp_path / "judges"
    write_judges(
        [_judge("fact-001", "vector", {"vsw": 0.5}, f=5, r=4, c=3)],
        judge_dir,
    )
    out = tmp_path / "metrics.csv"
    merge_to_csv(traces, judge_dir, out)

    with out.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_qa = {r["qa_id"]: r for r in rows}
    assert by_qa["fact-001"]["faithfulness"] == "5"
    assert by_qa["fact-001"]["relevance"] == "4"
    assert by_qa["fact-001"]["completeness"] == "3"
    assert by_qa["fact-001"]["faithfulness_reason"] == "fr"
    # Unjudged row → empty judge cells.
    assert by_qa["fact-002"]["faithfulness"] == ""
    assert by_qa["fact-002"]["relevance"] == ""
    assert by_qa["fact-002"]["completeness"] == ""


@pytest.mark.unit
def test_merge_to_csv_preserves_retrieval_metrics(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(
            qa_id="fact-001", mode="vector",
            # Use the real param keys that run_grid writes.
            params={
                "vector_similarity_weight": 0.5,
                "top_k": 5,
                "similarity_threshold": 0.1,
            },
            recall_at_k={"1": 1.0, "3": 1.0, "5": 1.0, "10": 1.0},
            mrr=1.0, ndcg=0.9,
        ),
    ])
    out = tmp_path / "metrics.csv"
    merge_to_csv(traces, tmp_path / "judges", out)
    with out.open("r", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["vsw"] == "0.5"
    assert row["top_k"] == "5"
    assert row["similarity_threshold"] == "0.1"
    assert row["recall_at_1"] == "1.0"
    assert row["mrr"] == "1.0"
    assert row["ndcg_at_10"] == "0.9"
    assert row["retrieve_es_ms"] == "2.0"
    assert row["generate_ms"] == "3.0"
    assert row["llm_calls"] == "1"


@pytest.mark.unit
def test_merge_to_csv_handles_empty_judge_dir(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
    ])
    out = tmp_path / "metrics.csv"
    n = merge_to_csv(traces, tmp_path / "judges_missing", out)
    assert n == 1
    with out.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["faithfulness"] == ""


# --------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_round_trip_read_write_merge(tmp_path):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="fact-002", mode="vector", params={"vsw": 0.5}),
        _trace_row(qa_id="refuse-001", mode="vector", params={"vsw": 0.5},
                   refusal_correct=True),
    ])
    judge_dir = tmp_path / "judges"

    # 1. Read all.
    batch1 = read_unjudged(traces, judge_dir, batch_size=2)
    assert len(batch1) == 2

    # 2. Score the batch and write it.
    scored = [_judge(b["qa_id"], b["mode"], b["params"], f=4, r=4, c=4)
              for b in batch1]
    write_judges(scored, judge_dir)

    # 3. Next read returns only the remaining row.
    batch2 = read_unjudged(traces, judge_dir, batch_size=10)
    assert len(batch2) == 1
    write_judges(
        [_judge(batch2[0]["qa_id"], batch2[0]["mode"], batch2[0]["params"],
                f=5, r=5, c=5)],
        judge_dir,
    )

    # 4. Now nothing left to judge.
    assert read_unjudged(traces, judge_dir, batch_size=10) == []

    # 5. Merge → all 3 rows with judge scores set.
    out = tmp_path / "metrics.csv"
    n = merge_to_csv(traces, judge_dir, out)
    assert n == 3
    with out.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["faithfulness"] in {"4", "5"} for r in rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_read_emits_json_array(tmp_path, capsys):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
    ])
    rc = _cli_main([
        "read", "--traces", str(traces),
        "--judge-dir", str(tmp_path / "judges"),
        "--batch-size", "10",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["qa_id"] == "fact-001"


@pytest.mark.unit
def test_cli_write_then_merge(tmp_path, capsys):
    _ds(tmp_path)
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, [
        _trace_row(qa_id="fact-001", mode="vector", params={"vsw": 0.5}),
    ])
    judge_dir = tmp_path / "judges"
    judges_file = tmp_path / "scored.json"
    judges_file.write_text(
        json.dumps([_judge("fact-001", "vector", {"vsw": 0.5})]),
        encoding="utf-8",
    )

    rc = _cli_main([
        "write", "--judges-json", str(judges_file),
        "--judge-dir", str(judge_dir),
    ])
    assert rc == 0
    # The CLI prints the written file path.
    out = capsys.readouterr().out.strip()
    assert out.endswith("batch_001.jsonl")

    out_csv = tmp_path / "metrics.csv"
    rc2 = _cli_main([
        "merge", "--traces", str(traces),
        "--judge-dir", str(judge_dir),
        "--out", str(out_csv),
    ])
    assert rc2 == 0
    assert out_csv.exists()


@pytest.mark.unit
def test_cli_write_rejects_non_array(tmp_path):
    judges_file = tmp_path / "scored.json"
    judges_file.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    rc = _cli_main([
        "write", "--judges-json", str(judges_file),
        "--judge-dir", str(tmp_path / "judges"),
    ])
    assert rc == 2
