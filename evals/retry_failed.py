"""Retry rows whose trace is null in a traces.jsonl.

Reuses run_grid._run_one to keep semantics identical. Supports concurrent
retry via ThreadPoolExecutor.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evals.run_grid import _run_one, _resolve_rag_binary, _fetch_chunk_contents
from evals.schema import load_dataset


def _retry_row(*, idx, qa, mode, params, top_k, kb, rag_bin):
    try:
        result = _run_one(
            rag_bin=rag_bin, qa=qa, mode=mode, params=params,
            kb=kb, top_k=top_k, runner=subprocess.run,
            fetcher=_fetch_chunk_contents,
        )
    except Exception as e:
        return idx, None, f"exception: {e}"

    if result.trace is None:
        return idx, None, "still failed"

    new_row = {
        "qa_id": result.qa_id,
        "mode": result.mode,
        "params": result.params,
        "trace": result.trace,
        "recall_at_k": {str(k): v for k, v in result.recall_at_k.items()},
        "mrr": result.mrr,
        "ndcg_at_10": result.ndcg_at_10,
        "refusal_correct": result.refusal_correct,
        "retrieved_contents": result.retrieved_contents,
    }
    return idx, new_row, f"OK (mrr={result.mrr:.3f})"


def main(traces_path: Path, dataset_path: Path, concurrency: int) -> None:
    rag_bin = _resolve_rag_binary()
    items_by_id = {qa.id: qa for qa in load_dataset(dataset_path)}

    with traces_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    tasks = []
    for idx, r in enumerate(rows):
        if r.get("trace") is not None:
            continue
        qa = items_by_id.get(r["qa_id"])
        if qa is None:
            continue
        params = dict(r["params"])
        top_k = int(params.pop("top_k", 5))
        kb = "test"
        tasks.append((idx, qa, r["mode"], params, top_k, kb))

    print(f"[INFO] {traces_path.name}: {len(tasks)} failed rows; concurrency={concurrency}", flush=True)

    recovered = 0
    still_failed = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_retry_row, idx=t[0], qa=t[1], mode=t[2], params=t[3],
                      top_k=t[4], kb=t[5], rag_bin=rag_bin): t[0]
            for t in tasks
        }
        for fut in as_completed(futures):
            completed += 1
            idx, new_row, msg = fut.result()
            if new_row is not None:
                rows[idx] = new_row
                recovered += 1
            else:
                still_failed += 1
            r0 = rows[idx]
            print(f"[{completed}/{len(tasks)}] qa={r0['qa_id']} mode={r0['mode']} -> {msg}", flush=True)

    with traces_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_ok = sum(1 for r in rows if r.get("trace") is not None)
    print(f"[INFO] {traces_path.name}: recovered={recovered} still_failed={still_failed} "
          f"final_ok={total_ok}/{len(rows)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("traces", type=Path)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--dataset", type=Path,
                   default=Path("/Users/leng/my-RAG/ragkit/evals/dataset.jsonl"))
    args = p.parse_args()
    main(args.traces, args.dataset, args.concurrency)
