"""Tests for ``evals.run_grid`` (P2.5: judge-free, concurrent, ES mget)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from typer.testing import CliRunner

from evals.run_grid import (
    _CONCURRENCY_DEFAULT,
    _CONCURRENCY_MAX,
    _CONCURRENCY_MIN,
    _CSV_COLUMNS,
    _clamp_concurrency,
    _fetch_chunk_contents,
    _resolve_top_k_dimension,
    _retrieved_contents_for_trace,
    app,
    build_ask_command,
    expand_grid,
    run,
)
from evals.schema import QAItem, save_dataset


@dataclass
class _Completed:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


def _dataset(tmp_path: Path) -> Path:
    items = [
        QAItem(id="fact-001", question="第一个问题？", category="factual",
               ground_truth_chunk_ids=["c1"], gold_answer="答案"),
        QAItem(id="fact-002", question="第二个问题？", category="factual",
               ground_truth_chunk_ids=["c2", "c3"], gold_answer="另一答案"),
        QAItem(id="refuse-001", question="不应回答的问题？", category="refusal",
               ground_truth_chunk_ids=[], gold_answer=None),
    ]
    path = tmp_path / "ds.jsonl"
    save_dataset(items, path)
    return path


def _config(dataset_path: Path) -> dict[str, Any]:
    return {
        "dataset": str(dataset_path),
        "output_dir": "unused-set-by-test",
        "modes": ["vector"],
        "kb": "test",
        "top_k": 5,
        "grid": {
            "vector_similarity_weight": [0.2, 0.5],
            "similarity_threshold": [0.1],
        },
    }


def _trace_payload(answer: str = "answer") -> dict[str, Any]:
    return {
        "question": "Q", "kb": "test", "mode": "vector", "top_k": 5,
        "level": None, "timestamp_iso": "2026-06-06T00:00:00+00:00",
        "retrieved": [
            {"chunk_id": "c1", "rank": 1, "score": 0.9, "kind": "chunk"},
        ],
        "timing": {
            "embed_ms": 1.0, "retrieve_es_ms": 2.0, "rerank_ms": 0.0,
            "generate_ms": 3.0, "total_ms": 6.0,
        },
        "cost": {
            "llm_calls": 1, "embedding_calls": 1,
            "tokens_in": 100, "tokens_out": 50, "est_cost_usd": 0.0001,
        },
        "params": {
            "vector_similarity_weight": 0.5, "similarity_threshold": 0.1,
            "chunk_token_num": 256, "local_top_k_seeds": 5,
            "local_top_k_text_units": 5, "local_top_k_communities": 5,
            "local_top_k_entities": 5, "local_top_k_relations": 5,
            "global_top_k_reports": 5, "map_batch_token_budget": 1000,
            "rating_threshold": 5, "default_final_top_k": 5,
        },
        "answer": answer,
    }


def _ok_runner(cmd, *, stdout=None, stderr=None, timeout=None):
    """Default fake runner: writes a successful trace to the --eval-out path."""
    idx = cmd.index("--eval-out")
    Path(cmd[idx + 1]).write_text(json.dumps(_trace_payload()), encoding="utf-8")
    return _Completed(returncode=0)


def _stub_fetcher(chunk_ids: list[str], kb: str) -> dict[str, str]:
    """Return canned content for any chunk id; no ES contact."""
    return {cid: f"<content-for {cid}>" for cid in chunk_ids}


# -------- grid expansion --------


@pytest.mark.unit
def test_expand_grid_cartesian():
    out = expand_grid({"a": [1, 2], "b": [3, 4]})
    assert {tuple(sorted(d.items())) for d in out} == {
        (("a", 1), ("b", 3)), (("a", 1), ("b", 4)),
        (("a", 2), ("b", 3)), (("a", 2), ("b", 4)),
    }


@pytest.mark.unit
def test_expand_grid_single_value_and_empty():
    assert expand_grid({"a": [1, 2], "b": [3]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
    assert expand_grid({}) == [{}]


# -------- command construction --------


@pytest.mark.unit
def test_build_ask_command_shape():
    cmd = build_ask_command(
        rag_bin="/usr/bin/rag", question="Q?", kb="test", mode="vector", top_k=5,
        params={"vector_similarity_weight": 0.5, "similarity_threshold": 0.1},
        eval_out="/tmp/trace.json",
    )
    assert cmd[:3] == ["/usr/bin/rag", "ask", "Q?"]
    for tok in ("--kb", "test", "--mode", "vector", "--top-k", "5",
                "--eval-trace", "--eval-out", "/tmp/trace.json",
                "--param", "vector_similarity_weight=0.5",
                "--param", "similarity_threshold=0.1"):
        assert tok in cmd


@pytest.mark.unit
def test_build_ask_command_param_order_deterministic():
    p = {"vector_similarity_weight": 0.5, "similarity_threshold": 0.1}
    a = build_ask_command(rag_bin="rag", question="Q", kb="k", mode="vector",
                          top_k=5, params=p, eval_out="/x")
    b = build_ask_command(rag_bin="rag", question="Q", kb="k", mode="vector",
                          top_k=5, params=dict(reversed(list(p.items()))),
                          eval_out="/x")
    assert a == b


# -------- top_k dimension resolution --------


@pytest.mark.unit
def test_resolve_top_k_uses_scalar_when_only_top_k_set():
    assert _resolve_top_k_dimension({"top_k": 5}) == [5]


@pytest.mark.unit
def test_resolve_top_k_uses_list_when_top_k_values_set():
    cfg = {"top_k": 5, "top_k_values": [3, 5, 10]}
    # top_k_values wins.
    assert _resolve_top_k_dimension(cfg) == [3, 5, 10]


@pytest.mark.unit
def test_resolve_top_k_empty_list_rejected():
    with pytest.raises(ValueError):
        _resolve_top_k_dimension({"top_k_values": []})


@pytest.mark.unit
def test_resolve_top_k_missing_both_rejected():
    with pytest.raises(ValueError):
        _resolve_top_k_dimension({"modes": ["vector"]})


# -------- concurrency clamping --------


@pytest.mark.unit
def test_clamp_concurrency_within_bounds():
    assert _clamp_concurrency(5) == 5
    assert _clamp_concurrency(_CONCURRENCY_MIN) == _CONCURRENCY_MIN
    assert _clamp_concurrency(_CONCURRENCY_MAX) == _CONCURRENCY_MAX


@pytest.mark.unit
def test_clamp_concurrency_clamps_low_and_high():
    assert _clamp_concurrency(0) == _CONCURRENCY_MIN
    assert _clamp_concurrency(-7) == _CONCURRENCY_MIN
    assert _clamp_concurrency(99) == _CONCURRENCY_MAX


# -------- _fetch_chunk_contents --------


@pytest.mark.unit
def test_fetch_chunk_contents_empty_input_short_circuits():
    # Empty input must NOT touch the ES client.
    assert _fetch_chunk_contents([], "test") == {}


@pytest.mark.unit
def test_fetch_chunk_contents_happy_path_with_mocked_es(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeES:
        def __init__(self, *a, **kw):
            captured["init_args"] = (a, kw)

        def mget(self, *, index, body, _source):
            captured["mget"] = {"index": index, "body": body, "_source": _source}
            return {"docs": [
                {"_id": "c1", "found": True,
                 "_source": {"content_with_weight": "alpha"}},
                {"_id": "c2", "found": True,
                 "_source": {"content_with_weight": "beta"}},
                {"_id": "c3", "found": False},
            ]}

    class _FakeCfg:
        es_host = "http://localhost:9200"
        es_user = "elastic"
        es_password = "pw"

    monkeypatch.setattr(
        "ragkit.config.get_config", lambda: _FakeCfg(), raising=True,
    )
    monkeypatch.setattr("elasticsearch.Elasticsearch", _FakeES, raising=True)

    out = _fetch_chunk_contents(["c1", "c2", "c3", "c1"], "test_kb")
    assert out == {"c1": "alpha", "c2": "beta"}
    # Dedup happened before mget.
    assert captured["mget"]["body"]["ids"] == ["c1", "c2", "c3"]
    assert captured["mget"]["index"] == "test_kb"


@pytest.mark.unit
def test_fetch_chunk_contents_es_failure_returns_empty(monkeypatch):
    class _FakeES:
        def __init__(self, *a, **kw): pass
        def mget(self, **kw):
            raise RuntimeError("connection refused")

    class _FakeCfg:
        es_host = "http://localhost:9200"
        es_user = "elastic"
        es_password = "pw"

    monkeypatch.setattr(
        "ragkit.config.get_config", lambda: _FakeCfg(), raising=True,
    )
    monkeypatch.setattr("elasticsearch.Elasticsearch", _FakeES, raising=True)

    assert _fetch_chunk_contents(["c1", "c2"], "kb") == {}


# -------- _retrieved_contents_for_trace --------


@pytest.mark.unit
def test_retrieved_contents_filters_non_chunk_kinds():
    trace = {"retrieved": [
        {"chunk_id": "c1", "kind": "chunk", "rank": 1, "score": 0.9},
        {"chunk_id": "e1", "kind": "entity", "rank": 2, "score": 0.8},
        {"chunk_id": "c2", "kind": "chunk", "rank": 3, "score": 0.7},
    ]}
    rc = _retrieved_contents_for_trace(trace, "kb", fetcher=_stub_fetcher)
    assert [r["chunk_id"] for r in rc] == ["c1", "c2"]
    assert all(r["content"].startswith("<content-for") for r in rc)


@pytest.mark.unit
def test_retrieved_contents_preserves_order():
    trace = {"retrieved": [
        {"chunk_id": "z", "kind": "chunk", "rank": 1, "score": 0.5},
        {"chunk_id": "a", "kind": "chunk", "rank": 2, "score": 0.4},
        {"chunk_id": "m", "kind": "chunk", "rank": 3, "score": 0.3},
    ]}
    rc = _retrieved_contents_for_trace(trace, "kb", fetcher=_stub_fetcher)
    assert [r["chunk_id"] for r in rc] == ["z", "a", "m"]


@pytest.mark.unit
def test_retrieved_contents_no_chunks_returns_empty():
    trace = {"retrieved": [
        {"chunk_id": "e1", "kind": "entity", "rank": 1, "score": 0.5},
    ]}
    rc = _retrieved_contents_for_trace(trace, "kb", fetcher=_stub_fetcher)
    assert rc == []


# -------- happy-path sweep (no judge) --------


@pytest.mark.unit
def test_run_full_sweep_with_mocks(tmp_path):
    cfg = _config(_dataset(tmp_path))
    out_dir = tmp_path / "out"
    captured: list[list[str]] = []
    lock_calls = []

    def runner(cmd, *, stdout=None, stderr=None, timeout=None):
        captured.append(list(cmd))
        lock_calls.append(cmd[cmd.index("--eval-out") + 1])
        return _ok_runner(cmd, stdout=stdout, stderr=stderr, timeout=timeout)

    returned = run(
        config=cfg, runner=runner, fetcher=_stub_fetcher,
        output_dir=out_dir, concurrency=1,
    )
    assert returned == out_dir

    # 3 qas × 1 mode × 2 grid combos = 6 runs.
    assert len(captured) == 6
    lines = (out_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    for ln in lines:
        row = json.loads(ln)
        for key in ("qa_id", "mode", "params", "trace",
                    "recall_at_k", "mrr", "ndcg_at_10",
                    "refusal_correct", "retrieved_contents"):
            assert key in row, f"missing key in row: {key}"
        # Hard requirement: no judge field on the row.
        assert "judge" not in row

    csv_text = (out_dir / "metrics.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0].split(",")
    assert header == list(_CSV_COLUMNS)
    # No judge dims in the CSV.
    for forbidden in ("faithfulness", "relevance", "completeness"):
        assert forbidden not in header
    assert len(csv_text.splitlines()) == 7  # header + 6 data rows


# -------- retrieved_contents in output rows --------


@pytest.mark.unit
def test_traces_jsonl_includes_retrieved_contents(tmp_path):
    cfg = _config(_dataset(tmp_path))
    out_dir = tmp_path / "out_rc"
    run(
        config=cfg, dry_run_n=1, runner=_ok_runner,
        fetcher=_stub_fetcher, output_dir=out_dir, concurrency=1,
    )
    rows = [json.loads(l) for l in
            (out_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        rc = row["retrieved_contents"]
        assert isinstance(rc, list)
        assert all("chunk_id" in r and "content" in r for r in rc)
        assert rc[0]["chunk_id"] == "c1"
        assert "<content-for c1>" == rc[0]["content"]


# -------- top_k_values produces a per-row --top-k --------


@pytest.mark.unit
def test_top_k_values_drives_separate_calls(tmp_path):
    cfg = _config(_dataset(tmp_path))
    cfg.pop("top_k")
    cfg["top_k_values"] = [3, 5, 10]
    cfg["grid"] = {
        "vector_similarity_weight": [0.5],
        "similarity_threshold": [0.1],
    }
    out_dir = tmp_path / "out_topk"
    captured: list[list[str]] = []

    def runner(cmd, *, stdout=None, stderr=None, timeout=None):
        captured.append(list(cmd))
        return _ok_runner(cmd, stdout=stdout, stderr=stderr, timeout=timeout)

    run(
        config=cfg, dry_run_n=1, runner=runner,
        fetcher=_stub_fetcher, output_dir=out_dir, concurrency=1,
    )
    # 1 qa × 1 mode × 1 combo × 3 top_k = 3 calls.
    assert len(captured) == 3
    seen_ks: set[str] = set()
    for cmd in captured:
        idx = cmd.index("--top-k")
        seen_ks.add(cmd[idx + 1])
    assert seen_ks == {"3", "5", "10"}


# -------- dry-run --------


@pytest.mark.unit
def test_dry_run_n_slices_dataset(tmp_path):
    cfg = _config(_dataset(tmp_path))
    out_dir = tmp_path / "out_dry"
    cmds: list[list[str]] = []

    def runner(cmd, *, stdout=None, stderr=None, timeout=None):
        cmds.append(list(cmd))
        return _ok_runner(cmd, stdout=stdout, stderr=stderr, timeout=timeout)

    run(
        config=cfg, dry_run_n=1, runner=runner,
        fetcher=_stub_fetcher, output_dir=out_dir, concurrency=1,
    )
    assert len(cmds) == 2  # 1 qa × 1 mode × 2 combos
    assert any("第一个问题？" in c for c in cmds)
    assert not any("第二个问题？" in c for c in cmds)


# -------- failure paths --------


@pytest.mark.unit
def test_subprocess_failure_doesnt_abort_sweep(tmp_path):
    cfg = _config(_dataset(tmp_path))
    cfg["grid"] = {"vector_similarity_weight": [0.5], "similarity_threshold": [0.1]}
    out_dir = tmp_path / "out_fail"

    def runner(cmd, *, stdout=None, stderr=None, timeout=None):
        if "第一个问题？" in cmd:
            return _Completed(returncode=1, stderr=b"boom")
        return _ok_runner(cmd, stdout=stdout, stderr=stderr, timeout=timeout)

    run(
        config=cfg, runner=runner, fetcher=_stub_fetcher,
        output_dir=out_dir, concurrency=1,
    )
    rows = [json.loads(l) for l in
            (out_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    failed = next(r for r in rows if r["qa_id"] == "fact-001")
    assert failed["recall_at_k"] in (None, {})
    assert failed["trace"] in (None, {})
    succeeded = [r for r in rows if r["qa_id"] != "fact-001"]
    assert all(r["trace"] for r in succeeded)


@pytest.mark.unit
def test_subprocess_retries_once_before_giving_up(tmp_path):
    cfg = _config(_dataset(tmp_path))
    cfg["grid"] = {"vector_similarity_weight": [0.5], "similarity_threshold": [0.1]}
    out_dir = tmp_path / "out_retry"
    attempts: list[int] = []

    def flaky(cmd, *, stdout=None, stderr=None, timeout=None):
        attempts.append(1)
        return _Completed(returncode=2, stderr=b"err")

    run(
        config=cfg, dry_run_n=1, runner=flaky,
        fetcher=_stub_fetcher, output_dir=out_dir, concurrency=1,
    )
    assert len(attempts) == 2  # original + one retry.


# -------- CLI: --concurrency surface --------


@pytest.mark.unit
def test_cli_help_shows_concurrency_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--concurrency" in out
    # Defaults are reflected.
    assert str(_CONCURRENCY_DEFAULT) in out
    # Old judge knob is gone.
    assert "--skip-judge" not in out


@pytest.mark.unit
def test_cli_concurrency_default_is_5(tmp_path):
    """Invoking CLI without --concurrency should pass 5 to run()."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "dataset: ds.jsonl\n"
        "output_dir: out\n"
        "modes: [vector]\n"
        "kb: test\n"
        "top_k: 5\n"
        "grid: {}\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return Path("/tmp/whatever")

    with mock.patch("evals.run_grid.run", side_effect=_fake_run):
        result = CliRunner().invoke(app, ["--config", str(cfg_yaml)])
    assert result.exit_code == 0
    assert captured["concurrency"] == _CONCURRENCY_DEFAULT


@pytest.mark.unit
def test_cli_concurrency_custom_value_passed_through(tmp_path):
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "dataset: ds.jsonl\n"
        "output_dir: out\n"
        "modes: [vector]\n"
        "kb: test\n"
        "top_k: 5\n"
        "grid: {}\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return Path("/tmp/whatever")

    with mock.patch("evals.run_grid.run", side_effect=_fake_run):
        result = CliRunner().invoke(
            app, ["--config", str(cfg_yaml), "--concurrency", "8"],
        )
    assert result.exit_code == 0
    assert captured["concurrency"] == 8


@pytest.mark.unit
def test_cli_concurrency_out_of_range_rejected(tmp_path):
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "dataset: ds.jsonl\n"
        "output_dir: out\n"
        "modes: [vector]\n"
        "kb: test\n"
        "top_k: 5\n"
        "grid: {}\n",
        encoding="utf-8",
    )
    # Typer min/max should reject 0 and 17 before run() is touched.
    for bad in ("0", "17"):
        result = CliRunner().invoke(
            app, ["--config", str(cfg_yaml), "--concurrency", bad],
        )
        assert result.exit_code != 0


# -------- CSV column stability --------


@pytest.mark.unit
def test_csv_columns_exact_order():
    """Pin _CSV_COLUMNS contract so downstream tooling doesn't silently break.

    P2.5: judge dims are NOT in this set — they're added by judge_helper merge.
    """
    assert _CSV_COLUMNS == (
        "qa_id", "category", "mode",
        "vsw", "top_k", "similarity_threshold",
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mrr", "ndcg_at_10", "refusal_correct",
        "retrieve_es_ms", "generate_ms", "total_ms",
        "llm_calls", "embedding_calls",
    )
