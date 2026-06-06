"""Sweep orchestrator: runs ``rag ask`` across (qa × mode × grid-combo).

For every combination it:
  1. Invokes ``rag ask --eval-trace --eval-out <tmpfile>`` as a subprocess.
  2. Reads the trace JSON.
  3. Computes retrieval metrics (see :mod:`evals.eval_retrieval`).
  4. Fetches the actual chunk content from ES for each retrieved chunk_id
     (fixes the P2 D4 limitation where chunk content wasn't available).
  5. Writes one ``SweepResultRow`` to ``traces.jsonl`` and one flat row to
     ``metrics.csv``.

P2.5 changes vs P2:
  - The LLM-as-judge step is gone. The human (Claude Opus 4.7 in the parent
    conversation) judges manually via ``evals/judge_helper.py``. Faithfulness,
    relevance, and completeness columns are added later by ``judge_helper
    merge``.
  - The per-(qa, params, mode) loop now runs under a
    ``ThreadPoolExecutor`` with configurable ``--concurrency``. Default 5,
    bounds [1, 16]. ES rerank is the bottleneck — > 16 in-flight rag asks
    saturates a single-node setup.
  - Each row now carries ``retrieved_contents``: ``{chunk_id, content}``
    pairs fetched from ES via ``mget``, so the human judge can read what
    the retriever actually surfaced.
  - The sweep config YAML supports either ``top_k`` (single int) or
    ``top_k_values`` (list of ints, each becomes a separate sweep dimension).

Failure handling: a failed subprocess is retried once; on second failure the
row is written with ``trace=null`` and empty metrics. ES ``mget`` failures
yield empty ``retrieved_contents`` but never abort the sweep.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from .eval_retrieval import compute_metrics
from .schema import QAItem, SweepResultRow, load_dataset


logger = logging.getLogger("evals.run_grid")
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# Fallback used only when ``shutil.which("rag")`` returns nothing.
_RAG_FALLBACK = "/Users/leng/miniforge3/envs/ragkit/bin/rag"

# Concurrency bounds for the --concurrency flag.
_CONCURRENCY_DEFAULT = 5
_CONCURRENCY_MIN = 1
_CONCURRENCY_MAX = 16

# Stable column order for downstream notebooks. Judge dims (faithfulness /
# relevance / completeness) are merged in later by ``judge_helper merge``.
_CSV_COLUMNS: tuple[str, ...] = (
    "qa_id", "category", "mode",
    "vsw", "top_k", "similarity_threshold",
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "mrr", "ndcg_at_10", "refusal_correct",
    "retrieve_es_ms", "generate_ms", "total_ms",
    "llm_calls", "embedding_calls",
)


def _resolve_rag_binary() -> str:
    found = shutil.which("rag")
    if found:
        return found
    logger.warning("`rag` not on PATH — using fallback %s", _RAG_FALLBACK)
    return _RAG_FALLBACK


def _resolve_output_dir(template: str) -> Path:
    return Path(template.format(timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")))


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of grid values. ``{}`` → ``[{}]``."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(grid[k] for k in keys))]


def build_ask_command(
    *, rag_bin: str, question: str, kb: str, mode: str, top_k: int,
    params: dict[str, Any], eval_out: str,
) -> list[str]:
    """Construct the ``rag ask ...`` argv. Public for tests."""
    cmd: list[str] = [
        rag_bin, "ask", question,
        "--kb", kb, "--mode", mode, "--top-k", str(top_k),
        "--eval-trace", "--eval-out", eval_out,
    ]
    # Sorted for determinism — same params → same command string.
    for key in sorted(params):
        cmd.extend(["--param", f"{key}={params[key]}"])
    return cmd


def _run_with_retry(
    cmd: list[str], *, runner: Any = subprocess.run, timeout: int = 600,
) -> bool:
    """Run ``cmd``; retry once on non-zero exit. Returns True iff success."""
    for attempt in (1, 2):
        try:
            done = runner(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)
        except Exception as e:
            logger.warning("attempt %d: subprocess raised %s", attempt, e)
            continue
        if done.returncode == 0:
            return True
        logger.warning("attempt %d: rag ask exited %s. stderr=%s",
                       attempt, done.returncode,
                       getattr(done, "stderr", b"")[-300:])
    return False


def _fetch_chunk_contents(chunk_ids: list[str], kb: str) -> dict[str, str]:
    """Return chunk_id -> content_with_weight mapping via ES mget.

    Empty dict on failure (logged, never raised). Deduplicates input ids
    while preserving order. Used to fix the P2 D4 limitation where the
    judge couldn't see chunk text — now the human judge gets full content
    alongside chunk_ids in ``traces.jsonl``.
    """
    if not chunk_ids:
        return {}
    from ragkit.config import get_config
    from elasticsearch import Elasticsearch
    cfg = get_config()
    es = Elasticsearch(
        [cfg.es_host],
        basic_auth=(cfg.es_user, cfg.es_password),
        verify_certs=False,
    )
    try:
        unique_ids = list(dict.fromkeys(chunk_ids))
        res = es.mget(
            index=kb,
            body={"ids": unique_ids},
            _source=["content_with_weight"],
        )
        out: dict[str, str] = {}
        for doc in res.get("docs", []):
            if doc.get("found"):
                out[doc["_id"]] = (doc.get("_source") or {}).get(
                    "content_with_weight", ""
                )
        return out
    except Exception as e:
        # Don't abort sweep on ES hiccup; log and return what we have.
        logger.warning("ES mget failed for %d chunks (kb=%s): %s",
                       len(chunk_ids), kb, e)
        return {}


def _retrieved_contents_for_trace(
    trace: dict, kb: str,
    *, fetcher: Any = _fetch_chunk_contents,
) -> list[dict[str, str]]:
    """Build the ``retrieved_contents`` list for one trace.

    Only items with ``kind == "chunk"`` are fetched; entities, communities,
    relations have no chunk content. Order matches ``trace.retrieved`` order.
    """
    chunk_ids: list[str] = []
    for item in trace.get("retrieved") or []:
        if item.get("kind", "chunk") == "chunk":
            chunk_ids.append(item["chunk_id"])
    if not chunk_ids:
        return []
    id_to_content = fetcher(chunk_ids, kb)
    return [
        {"chunk_id": cid, "content": id_to_content.get(cid, "")}
        for cid in chunk_ids
    ]


def _run_one(
    *, rag_bin: str, qa: QAItem, mode: str, params: dict[str, Any],
    kb: str, top_k: int, runner: Any,
    fetcher: Any = _fetch_chunk_contents,
) -> SweepResultRow:
    """Execute one (qa, mode, params). Always returns a row (failures → empty)."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False, encoding="utf-8",
    ) as tmp:
        tmp_path = tmp.name

    try:
        cmd = build_ask_command(
            rag_bin=rag_bin, question=qa.question, kb=kb, mode=mode,
            top_k=top_k, params=params, eval_out=tmp_path,
        )
        ok = _run_with_retry(cmd, runner=runner)

        trace: dict[str, Any] | None = None
        if ok:
            try:
                trace = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("qa=%s mode=%s: trace read failed: %s",
                               qa.id, mode, e)

        metrics_dict: dict[str, float] | None = None
        if trace:
            try:
                metrics_dict = dict(compute_metrics(trace, qa))
            except Exception as e:
                logger.warning("qa=%s mode=%s: compute_metrics raised %s",
                               qa.id, mode, e)

        if metrics_dict is None:
            recall_at_k: dict[int, float] = {}
            mrr_val = 0.0
            ndcg_val = 0.0
            refusal_correct = False
        else:
            recall_at_k = {k: metrics_dict[f"recall_at_{k}"] for k in (1, 3, 5, 10)}
            mrr_val = metrics_dict["mrr"]
            ndcg_val = metrics_dict["ndcg_at_10"]
            refusal_correct = bool(metrics_dict.get("refusal_correct", False))

        # P2.5 D4 fix: fetch real chunk content for each retrieved item.
        retrieved_contents: list[dict[str, str]] = []
        if trace:
            try:
                retrieved_contents = _retrieved_contents_for_trace(
                    trace, kb, fetcher=fetcher,
                )
            except Exception as e:
                logger.warning("qa=%s mode=%s: chunk content fetch raised %s",
                               qa.id, mode, e)

        # Add top_k into params so the CSV row can record it even when
        # ``top_k_values`` drives a per-row k.
        merged_params = {**params, "top_k": top_k}

        return SweepResultRow(
            qa_id=qa.id, mode=mode, params=merged_params,
            trace=trace,
            recall_at_k=recall_at_k, mrr=mrr_val, ndcg_at_10=ndcg_val,
            refusal_correct=refusal_correct,
            retrieved_contents=retrieved_contents,
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def _row_to_csv_dict(row: SweepResultRow, qa: QAItem) -> dict[str, Any]:
    p = row.params
    timing = (row.trace or {}).get("timing") or {}
    cost = (row.trace or {}).get("cost") or {}
    return {
        "qa_id": row.qa_id, "category": qa.category, "mode": row.mode,
        "vsw": p.get("vector_similarity_weight"),
        "top_k": p.get("top_k"),
        "similarity_threshold": p.get("similarity_threshold"),
        "recall_at_1": row.recall_at_k.get(1, ""),
        "recall_at_3": row.recall_at_k.get(3, ""),
        "recall_at_5": row.recall_at_k.get(5, ""),
        "recall_at_10": row.recall_at_k.get(10, ""),
        "mrr": row.mrr, "ndcg_at_10": row.ndcg_at_10,
        "refusal_correct": row.refusal_correct,
        "retrieve_es_ms": timing.get("retrieve_es_ms", ""),
        "generate_ms": timing.get("generate_ms", ""),
        "total_ms": timing.get("total_ms", ""),
        "llm_calls": cost.get("llm_calls", ""),
        "embedding_calls": cost.get("embedding_calls", ""),
    }


def _resolve_top_k_dimension(config: dict[str, Any]) -> list[int]:
    """Return the list of top_k values to sweep over.

    Precedence: ``top_k_values`` (list, sweep dim) wins over ``top_k``
    (single int). If neither is present, raise.
    """
    if "top_k_values" in config and config["top_k_values"] is not None:
        vals = list(config["top_k_values"])
        if not vals:
            raise ValueError("top_k_values must be a non-empty list")
        return [int(v) for v in vals]
    if "top_k" in config and config["top_k"] is not None:
        return [int(config["top_k"])]
    raise ValueError("config must define either top_k or top_k_values")


def _clamp_concurrency(n: int) -> int:
    return max(_CONCURRENCY_MIN, min(_CONCURRENCY_MAX, int(n)))


def run(
    *, config: dict[str, Any], dry_run_n: int | None = None,
    runner: Any = subprocess.run,
    fetcher: Any = _fetch_chunk_contents,
    output_dir: Path | None = None,
    concurrency: int = _CONCURRENCY_DEFAULT,
) -> Path:
    """Execute a sweep. Returns the output directory.

    Workers run under a thread pool of size ``concurrency``. Results are
    collected ``as_completed`` and written to traces.jsonl + metrics.csv
    under a writer lock so the files stay valid JSONL/CSV.
    """
    qas = load_dataset(Path(config["dataset"]))
    if dry_run_n is not None:
        qas = qas[:dry_run_n]

    out_dir = output_dir if output_dir is not None else _resolve_output_dir(
        config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    csv_path = out_dir / "metrics.csv"

    combos = expand_grid(config.get("grid") or {})
    modes = list(config["modes"])
    kb = config["kb"]
    top_ks = _resolve_top_k_dimension(config)
    rag_bin = _resolve_rag_binary()
    cc = _clamp_concurrency(concurrency)
    total = len(qas) * len(modes) * len(combos) * len(top_ks)

    logger.info(
        "sweep: %d qa × %d modes × %d combos × %d top_k = %d runs → %s "
        "(concurrency=%d)",
        len(qas), len(modes), len(combos), len(top_ks), total, out_dir, cc,
    )

    # Build the full job list up front so as_completed ordering is independent
    # of submission order. (qa, mode, params, top_k) tuples.
    jobs: list[tuple[QAItem, str, dict[str, Any], int]] = []
    for qa in qas:
        for mode in modes:
            for params in combos:
                for k in top_ks:
                    jobs.append((qa, mode, params, k))

    writer_lock = threading.Lock()
    counter = 0

    with traces_path.open("w", encoding="utf-8") as traces_fh, \
            csv_path.open("w", encoding="utf-8", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        csv_fh.flush()

        def _do(job: tuple[QAItem, str, dict[str, Any], int]) -> tuple[QAItem, SweepResultRow]:
            qa, mode, params, k = job
            row = _run_one(
                rag_bin=rag_bin, qa=qa, mode=mode, params=params,
                kb=kb, top_k=k, runner=runner, fetcher=fetcher,
            )
            return qa, row

        with ThreadPoolExecutor(max_workers=cc) as pool:
            futures = [pool.submit(_do, job) for job in jobs]
            for fut in as_completed(futures):
                try:
                    qa, row = fut.result()
                except Exception as e:
                    # Defensive: _do should never raise (failures yield empty
                    # rows inside _run_one). If it does, log and skip.
                    logger.error("worker raised unexpectedly: %s", e)
                    continue
                with writer_lock:
                    counter += 1
                    traces_fh.write(
                        json.dumps(asdict(row), ensure_ascii=False) + "\n"
                    )
                    traces_fh.flush()
                    writer.writerow(_row_to_csv_dict(row, qa))
                    csv_fh.flush()
                    print(
                        f"[{counter}/{total}] qa={row.qa_id} mode={row.mode} "
                        f"params={row.params} mrr={row.mrr:.3f} "
                        f"ndcg={row.ndcg_at_10:.3f}",
                        flush=True,
                    )
    logger.info("sweep done: %s", out_dir)
    return out_dir


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    config: Path = typer.Option(
        ..., "--config", "-c",
        help="Path to a sweep YAML (e.g. evals/sweep_e1_vsw.yaml).",
    ),
    dry_run_n: int = typer.Option(
        None, "--dry-run-n",
        help="Process only the first N dataset items (smoke).",
    ),
    concurrency: int = typer.Option(
        _CONCURRENCY_DEFAULT, "--concurrency",
        min=_CONCURRENCY_MIN, max=_CONCURRENCY_MAX,
        help=(
            f"Parallel rag-ask workers. Default {_CONCURRENCY_DEFAULT}, "
            f"bounds [{_CONCURRENCY_MIN}, {_CONCURRENCY_MAX}]."
        ),
    ),
) -> None:
    """Run a sweep defined by ``--config``."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    run(config=cfg, dry_run_n=dry_run_n, concurrency=concurrency)


if __name__ == "__main__":
    app()
