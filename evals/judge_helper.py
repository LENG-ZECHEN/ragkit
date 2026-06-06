"""Human-in-the-loop judging helper for the P2.5 eval workflow.

The P2 LLM-as-judge step has been retired. Judging is now done by the human
(Claude Opus 4.7 in the parent conversation), using a simple three-step
``read → judge → write → merge`` loop:

  1. ``read_unjudged`` pulls up to N untouched (qa, mode, params) rows from
     ``traces.jsonl`` and emits them as a JSON batch. The human reads the
     ``generated_answer`` + ``retrieved_contents`` and assigns 1-5 scores
     for faithfulness / relevance / completeness, plus a short reason for
     each.

  2. The human pastes their scored batch back as a JSON file.

  3. ``write_judges`` appends that batch to ``judge_dir/batch_NNN.jsonl``
     with auto-increment N. Scores must be int in [1, 5]; invalid input
     is rejected with ``ValueError``.

  4. ``merge_to_csv`` joins ``traces.jsonl`` + every batch_*.jsonl into a
     flat metrics.csv with judge columns populated. Rows still unjudged
     get empty faith/rel/comp cells.

Why no LLM judge?
  - qwen-plus (our generation model) exhibits self-preference bias when
    used as its own judge.
  - Opus 4.7 in-conversation is free, stronger, and already in the loop.

Join key for traces ↔ judges: ``(qa_id, mode, sorted_params_json)``. We
re-serialize params with sorted keys so a dict written by run_grid in one
order matches a judge dict edited by hand in another order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from .schema import load_dataset


_BATCH_RE = re.compile(r"^batch_(\d+)\.jsonl$")

_SCORE_KEYS: tuple[str, ...] = ("faithfulness", "relevance", "completeness")
_REASON_KEYS: tuple[str, ...] = (
    "faithfulness_reason", "relevance_reason", "completeness_reason",
)
_MIN_SCORE = 1
_MAX_SCORE = 5

# CSV columns produced by ``merge_to_csv``. The retrieval columns mirror
# run_grid._CSV_COLUMNS but interleave the judge dims.
_MERGED_COLUMNS: tuple[str, ...] = (
    "qa_id", "category", "mode",
    "vsw", "top_k", "similarity_threshold",
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "mrr", "ndcg_at_10", "refusal_correct",
    "faithfulness", "relevance", "completeness",
    "faithfulness_reason", "relevance_reason", "completeness_reason",
    "retrieve_es_ms", "generate_ms", "total_ms",
    "llm_calls", "embedding_calls",
)


def _params_key(params: dict[str, Any]) -> str:
    """Canonical join key for a params dict: JSON with sorted keys."""
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def _row_key(qa_id: str, mode: str, params: dict[str, Any]) -> tuple[str, str, str]:
    return (qa_id, mode, _params_key(params))


def _iter_jsonl(path: Path):
    """Yield JSON objects line-by-line, skipping blanks."""
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def _list_judge_files(judge_dir: Path) -> list[Path]:
    """Return all ``batch_NNN.jsonl`` files in ``judge_dir`` sorted by N."""
    if not judge_dir.exists():
        return []
    files: list[tuple[int, Path]] = []
    for p in judge_dir.iterdir():
        m = _BATCH_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return [p for _, p in sorted(files, key=lambda t: t[0])]


def _existing_judge_keys(judge_dir: Path) -> set[tuple[str, str, str]]:
    """Set of (qa_id, mode, params_key) tuples already judged."""
    keys: set[tuple[str, str, str]] = set()
    for jf in _list_judge_files(judge_dir):
        for obj in _iter_jsonl(jf):
            keys.add(_row_key(obj["qa_id"], obj["mode"], obj.get("params") or {}))
    return keys


def _next_batch_number(judge_dir: Path) -> int:
    """Next free batch number, starting at 1."""
    files = _list_judge_files(judge_dir)
    if not files:
        return 1
    max_n = 0
    for p in files:
        m = _BATCH_RE.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _validate_judge(j: dict[str, Any]) -> None:
    """Raise ValueError on a malformed judge dict.

    Required keys: ``qa_id``, ``mode``, ``params``, three score+reason pairs.
    Scores must be integer-valued in ``[1, 5]``.
    """
    for k in ("qa_id", "mode", "params"):
        if k not in j:
            raise ValueError(f"judge missing required field: {k}")
    if not isinstance(j["params"], dict):
        raise ValueError("judge.params must be a dict")
    for sk in _SCORE_KEYS:
        if sk not in j:
            raise ValueError(f"judge missing score field: {sk}")
        v = j[sk]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"judge.{sk} must be an int in [{_MIN_SCORE}, {_MAX_SCORE}], "
                f"got {v!r}"
            )
        if not (_MIN_SCORE <= v <= _MAX_SCORE):
            raise ValueError(
                f"judge.{sk} = {v} out of range [{_MIN_SCORE}, {_MAX_SCORE}]"
            )
    for rk in _REASON_KEYS:
        if rk not in j:
            raise ValueError(f"judge missing reason field: {rk}")
        if not isinstance(j[rk], str):
            raise ValueError(f"judge.{rk} must be a string")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def read_unjudged(
    traces_path: Path, judge_dir: Path, batch_size: int = 30,
) -> list[dict[str, Any]]:
    """Read up to ``batch_size`` unjudged trace rows from ``traces.jsonl``.

    A row is 'judged' if its ``(qa_id, mode, sorted_params)`` key appears in
    any ``judge_dir/batch_*.jsonl``. The returned batch is deterministically
    sorted by ``(mode, qa_id, sorted_params)`` so re-runs pick the same items.

    Each returned dict carries the fields the human judge needs:
        - qa_id, category, question, gold_answer
        - mode, params
        - generated_answer, retrieved_contents

    ``category``, ``question``, and ``gold_answer`` are looked up from the
    sibling ``dataset.jsonl`` if locatable, falling back to empty strings.
    """
    if batch_size <= 0:
        return []

    judged = _existing_judge_keys(judge_dir)

    # Best-effort dataset lookup: traces.jsonl doesn't carry question text.
    qa_lookup: dict[str, Any] = {}
    # Common case: traces.jsonl lives under evals/results/.../some_dir/,
    # so we search a few parent levels for dataset.jsonl.
    for candidate in (
        traces_path.parent / "dataset.jsonl",
        traces_path.parent.parent / "dataset.jsonl",
        traces_path.parent.parent.parent / "dataset.jsonl",
        Path("evals/dataset.jsonl"),
    ):
        if candidate.exists():
            try:
                for qa in load_dataset(candidate):
                    qa_lookup[qa.id] = qa
            except Exception:
                pass
            break

    candidates: list[dict[str, Any]] = []
    for row in _iter_jsonl(traces_path):
        key = _row_key(row["qa_id"], row["mode"], row.get("params") or {})
        if key in judged:
            continue
        qa = qa_lookup.get(row["qa_id"])
        trace = row.get("trace") or {}
        candidates.append({
            "qa_id": row["qa_id"],
            "category": getattr(qa, "category", "") if qa else "",
            "question": getattr(qa, "question", "") if qa else "",
            "gold_answer": getattr(qa, "gold_answer", None) if qa else None,
            "mode": row["mode"],
            "params": row.get("params") or {},
            "generated_answer": trace.get("answer", ""),
            "retrieved_contents": row.get("retrieved_contents") or [],
        })

    candidates.sort(key=lambda r: (r["mode"], r["qa_id"], _params_key(r["params"])))
    return candidates[:batch_size]


def write_judges(judges: list[dict[str, Any]], judge_dir: Path) -> Path:
    """Append a list of human judgements as a new ``batch_NNN.jsonl`` file.

    Each judge dict must include ``qa_id``, ``mode``, ``params``, the three
    score keys (int 1-5), and the three reason keys (str). Raises
    ``ValueError`` on any invalid entry — nothing is written.
    """
    if not judges:
        raise ValueError("write_judges: empty batch refused")
    for i, j in enumerate(judges):
        try:
            _validate_judge(j)
        except ValueError as e:
            raise ValueError(f"judge[{i}]: {e}") from None

    judge_dir.mkdir(parents=True, exist_ok=True)
    n = _next_batch_number(judge_dir)
    out_path = judge_dir / f"batch_{n:03d}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for j in judges:
            fh.write(json.dumps(j, ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def merge_to_csv(
    traces_path: Path, judge_dir: Path, metrics_csv: Path,
) -> int:
    """Merge traces.jsonl + all batch_*.jsonl into a flat metrics.csv.

    Joins on (qa_id, mode, sorted_params). Rows with no matching judge get
    empty faith/rel/comp/reason cells. ``category`` is best-effort sourced
    from the dataset sibling to traces.jsonl. Returns the number of data
    rows written (excludes the header).
    """
    judge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for jf in _list_judge_files(judge_dir):
        for obj in _iter_jsonl(jf):
            key = _row_key(obj["qa_id"], obj["mode"], obj.get("params") or {})
            judge_map[key] = obj  # later batches override earlier ones.

    qa_lookup: dict[str, Any] = {}
    for candidate in (
        traces_path.parent / "dataset.jsonl",
        traces_path.parent.parent / "dataset.jsonl",
        traces_path.parent.parent.parent / "dataset.jsonl",
        Path("evals/dataset.jsonl"),
    ):
        if candidate.exists():
            try:
                for qa in load_dataset(candidate):
                    qa_lookup[qa.id] = qa
            except Exception:
                pass
            break

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with metrics_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_MERGED_COLUMNS))
        writer.writeheader()
        for row in _iter_jsonl(traces_path):
            params = row.get("params") or {}
            trace = row.get("trace") or {}
            timing = trace.get("timing") or {}
            cost = trace.get("cost") or {}
            recall = row.get("recall_at_k") or {}

            qa = qa_lookup.get(row["qa_id"])
            category = getattr(qa, "category", "") if qa else ""

            key = _row_key(row["qa_id"], row["mode"], params)
            judge = judge_map.get(key) or {}

            out_row: dict[str, Any] = {
                "qa_id": row["qa_id"],
                "category": category,
                "mode": row["mode"],
                "vsw": params.get("vector_similarity_weight", ""),
                "top_k": params.get("top_k", ""),
                "similarity_threshold": params.get("similarity_threshold", ""),
                "recall_at_1": recall.get("1", recall.get(1, "")),
                "recall_at_3": recall.get("3", recall.get(3, "")),
                "recall_at_5": recall.get("5", recall.get(5, "")),
                "recall_at_10": recall.get("10", recall.get(10, "")),
                "mrr": row.get("mrr", ""),
                "ndcg_at_10": row.get("ndcg_at_10", ""),
                "refusal_correct": row.get("refusal_correct", ""),
                "faithfulness": judge.get("faithfulness", ""),
                "relevance": judge.get("relevance", ""),
                "completeness": judge.get("completeness", ""),
                "faithfulness_reason": judge.get("faithfulness_reason", ""),
                "relevance_reason": judge.get("relevance_reason", ""),
                "completeness_reason": judge.get("completeness_reason", ""),
                "retrieve_es_ms": timing.get("retrieve_es_ms", ""),
                "generate_ms": timing.get("generate_ms", ""),
                "total_ms": timing.get("total_ms", ""),
                "llm_calls": cost.get("llm_calls", ""),
                "embedding_calls": cost.get("embedding_calls", ""),
            }
            writer.writerow(out_row)
            count += 1
    return count


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.judge_helper",
        description="Human-in-the-loop judging helper (read / write / merge).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read", help="Print unjudged batch as JSON.")
    p_read.add_argument("--traces", required=True, type=Path)
    p_read.add_argument("--judge-dir", required=True, type=Path)
    p_read.add_argument("--batch-size", type=int, default=30)

    p_write = sub.add_parser("write", help="Append a judgement batch.")
    p_write.add_argument("--judges-json", required=True, type=Path)
    p_write.add_argument("--judge-dir", required=True, type=Path)

    p_merge = sub.add_parser("merge", help="Merge traces + judges into a CSV.")
    p_merge.add_argument("--traces", required=True, type=Path)
    p_merge.add_argument("--judge-dir", required=True, type=Path)
    p_merge.add_argument("--out", required=True, type=Path)

    return parser


def _cli_main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "read":
        batch = read_unjudged(args.traces, args.judge_dir, args.batch_size)
        print(json.dumps(batch, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "write":
        data = json.loads(args.judges_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print("ERROR: --judges-json must contain a JSON array", file=sys.stderr)
            return 2
        path = write_judges(data, args.judge_dir)
        print(str(path))
        return 0
    if args.cmd == "merge":
        n = merge_to_csv(args.traces, args.judge_dir, args.out)
        print(f"merged {n} rows → {args.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli_main())
