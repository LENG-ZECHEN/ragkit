# ragkit Evaluation Harness

This directory holds the **dataset, schemas, sweep configurations, and the
human-in-the-loop judging helper** that the Phase 2 evaluation scripts use.

The Phase 0 instrumentation layer (commit `519bf25`) added the
`--eval-trace` / `--param key=value` flags to `rag ask`; this harness builds on
the `EvalTrace` JSON those flags emit. See
[`src/ragkit/eval_context.py`](../src/ragkit/eval_context.py) for the trace
schema.

P2.5 retired the LLM-as-judge module — see **Judge** below.

---

## Directory layout

```
evals/
├── README.md                 # this file
├── __init__.py               # package marker
├── schema.py                 # QAItem, JudgeScores, SweepResultRow + JSONL I/O
├── judge_prompts.py          # Chinese rubric text reused by the human judge
├── judge_helper.py           # read / write / merge for human judging
├── eval_retrieval.py         # Recall@K / MRR / nDCG@10 computation
├── run_grid.py               # sweep orchestrator (concurrency, ES mget, judge-free)
├── sweep_e1_vsw.yaml         # E1: vector_similarity_weight grid
├── sweep_e2_topk.yaml        # E2: top_k diminishing-return curve
├── dataset.example.jsonl     # 3 placeholder rows showing the dataset shape
├── dataset.jsonl             # YOUR labeled QA pairs (gitignored / curated)
└── results/                  # populated by sweep runs
    └── {timestamp}/
        └── {experiment}/
            ├── traces.jsonl  # one SweepResultRow per (qa × mode × params) combo
            ├── metrics.csv   # flat retrieval-only metrics (judge cols added later)
            └── judges/       # human judgements: batch_001.jsonl, batch_002.jsonl, ...
```

---

## Dataset format (JSONL)

One JSON object per line. Field meanings (see `schema.py :: QAItem`):

| Field                     | Type             | Required | Meaning                                                                                                  |
| ------------------------- | ---------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `id`                      | `str`            | yes      | Stable identifier, e.g. `fact-001`. Reused across sweep CSVs so rows can be joined.                      |
| `question`                | `str`            | yes      | The user-facing question, in Chinese.                                                                    |
| `category`                | `str`            | yes      | One of `factual`, `passage_quoted`, `cross_paragraph_theme`, `refusal`.                                  |
| `ground_truth_chunk_ids`  | `list[str]`      | yes      | Chunk IDs that SHOULD appear in retrieval. Used for Recall@k / MRR / nDCG. Empty list for refusal-cases. |
| `gold_answer`             | `str` or `null`  | yes      | Reference answer text, or `null` for refusal-cases.                                                      |
| `notes`                   | `str` or `null`  | no       | Free-text annotator notes; ignored by metrics.                                                           |

Target distribution (~20 rows):

| Category                | Count | What it tests                                          |
| ----------------------- | ----- | ------------------------------------------------------ |
| `factual`               | 6     | Single fact recall (date, name, number)                |
| `passage_quoted`        | 6     | Verbatim quote of a specific passage                   |
| `cross_paragraph_theme` | 4     | Thematic summary across multiple paragraphs            |
| `refusal`               | 2     | Out-of-scope question — system should decline to guess |

---

## Question categories — definitions and expected winners

| Category                | Definition                                                                                              | Expected winner mode |
| ----------------------- | ------------------------------------------------------------------------------------------------------- | -------------------- |
| `factual`               | A single fact lives in a single chunk; retrieval just needs to find it.                                 | `vector`             |
| `passage_quoted`        | The answer must reproduce a specific passage; lexical match dominates.                                  | `vector` (high `vector_similarity_weight`) |
| `cross_paragraph_theme` | The answer synthesizes information distributed across paragraphs / entities.                            | `local`              |
| `refusal`               | The KB does NOT contain the answer; the system should decline rather than hallucinate.                  | `global` (least likely to over-retrieve) |

The sweeps below are designed to surface these "winner" intuitions
empirically rather than assume them.

---

## Experiments

The harness ships two experiment configs. **E3 (rerank toggle)** is in
"Future Work" — it requires a Dealer-level change to the rag pipeline and
is deferred.

### E1 — Vector similarity weight (`sweep_e1_vsw.yaml`)

Sweeps `vector_similarity_weight ∈ {0.0, 0.3, 0.5, 0.7, 0.95}` over all three
modes. Goal: find the BM25 ↔ Dense mixing point that maximises mean nDCG@10
on this corpus. Expected shape: U-curve with `0.95` strong on `passage_quoted`
and a mid-range value (0.3-0.5) strong on `factual` / `cross_paragraph_theme`.

  - Plan: 5 vsw × 3 modes × 20 questions = **300** `rag ask` calls
  - Estimated runtime: ~10 min @ `--concurrency 5`
  - Estimated cost: ¥3-5 (DashScope qwen-plus generation)

### E2 — Top-K (`sweep_e2_topk.yaml`)

Sweeps `top_k ∈ {3, 5, 10}` at a fixed mid-range `vsw=0.6`. Goal: confirm
the diminishing-return curve and pick the lowest k where Recall@k plateaus.
Useful for cost-tuning generation.

  - Plan: 3 top_k × 3 modes × 20 questions = **180** `rag ask` calls
  - Estimated runtime: ~6 min @ `--concurrency 5`
  - Estimated cost: ¥1-2

### E3 — Rerank toggle (Future Work)

Toggling the ES rerank stage on/off would isolate its contribution to
retrieval quality, but the rerank step is currently baked into the Dealer
and there's no `--param` knob for it. Adding one is out of scope for P2.5.

---

## Running a sweep

```bash
# 1. Curate your dataset (one-time).
cp evals/dataset.example.jsonl evals/dataset.jsonl
# ...edit dataset.jsonl to ~20 real labeled rows...

# 2. Run an experiment.
python -m evals.run_grid --config evals/sweep_e1_vsw.yaml --concurrency 5

# 3. Inspect the dry-run-friendly metrics first.
python -m evals.run_grid --config evals/sweep_e1_vsw.yaml --dry-run-n 2
```

Output goes to `evals/results/{timestamp}/{experiment}/`:
  - `traces.jsonl` — one full `SweepResultRow` per line, including the
    fetched `retrieved_contents` (chunk_id + content from ES).
  - `metrics.csv` — flat retrieval-only metrics ready for pandas / a notebook.
    Judge columns (faithfulness/relevance/completeness) are added by the
    `merge` step below.

---

## Judge — human-in-the-loop (P2.5 onwards)

The judge is **Claude Opus 4.7 in the conversation that orchestrated this
sweep** — i.e. me, reading the trace and assigning scores by hand.

### Why no LLM judge?

The earlier P2 LLM-as-judge module used `qwen-plus` against its own
generations, which exhibits self-preference bias (a model that generated
the answer scores it higher than a different model would). Opus 4.7 in the
parent conversation is both free (no extra API spend) and a stronger judge
than qwen-plus, so the LLM judge was retired.

### Judge rubric (1-5, three dimensions)

| Dimension           | 5 (best)                                                  | 3                                            | 1 (worst)                                   |
| ------------------- | --------------------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| **Faithfulness**    | Every claim aligns with the retrieved context line-by-line | Main claims supported, minor unsupported details | Contradicts or fabricates with no support  |
| **Relevance**       | Directly answers the question, no drift                    | Partial answer mixed with off-topic content  | Answers a different question                |
| **Completeness**    | Covers all key facts in the gold answer                    | Covers about half                            | Misses almost all key facts                 |

Refusal-case rule: if `gold_answer` is `null`, a correct refusal scores **5
on all three dimensions**; a hallucinated answer scores **1 on faithfulness**.

The full Chinese rubric text lives in
[`judge_prompts.py`](./judge_prompts.py) and remains the canonical source.

### Read → judge → write → merge loop

```bash
# Resolve paths once.
TRACES=evals/results/20260606_120000/e1_vsw/traces.jsonl
JUDGES=evals/results/20260606_120000/e1_vsw/judges

# 1. Pull up to 30 unjudged rows as a JSON batch.
python -m evals.judge_helper read \
    --traces $TRACES --judge-dir $JUDGES --batch-size 30 > batch.json

# 2. The human (me, Opus 4.7 in the conversation) reads batch.json, scores
#    every row, and writes the scored array to scored.json with shape:
#    [{"qa_id": "...", "mode": "...", "params": {...},
#      "faithfulness": 4, "faithfulness_reason": "...",
#      "relevance": 5, "relevance_reason": "...",
#      "completeness": 3, "completeness_reason": "..."}, ...]

# 3. Append the batch to the judge dir as batch_NNN.jsonl.
python -m evals.judge_helper write \
    --judges-json scored.json --judge-dir $JUDGES

# 4. Repeat 1-3 until `read` returns []. Then merge.
python -m evals.judge_helper merge \
    --traces $TRACES --judge-dir $JUDGES \
    --out evals/results/20260606_120000/e1_vsw/metrics_with_judges.csv
```

`batch_NNN` is auto-incremented; the merge step joins on
`(qa_id, mode, sorted_params)` so order of batches doesn't matter.

### Output contract for one judge row

```json
{
  "qa_id": "fact-001",
  "mode": "vector",
  "params": {"vector_similarity_weight": 0.5, "similarity_threshold": 0.1, "top_k": 5},
  "faithfulness": 4, "faithfulness_reason": "minor unsupported numeric claim",
  "relevance":    5, "relevance_reason":    "directly addresses the question",
  "completeness": 3, "completeness_reason": "missing one of two key dates"
}
```

Scores must be integers in `[1, 5]`; reasons must be strings. `write_judges`
rejects malformed input with `ValueError`.

---

## Interpreting results

Sweep output lands in `evals/results/{timestamp}/{experiment}/`:

| Artifact                   | Contents                                                                                                                |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `traces.jsonl`             | One `SweepResultRow` per line — params, full `EvalTrace`, retrieval metrics, and the chunk content the human judge sees. |
| `metrics.csv`              | Retrieval-only flat table — Recall@k / MRR / nDCG@10 / refusal_correct / timing / cost. No judge dims.                 |
| `judges/batch_*.jsonl`     | Human judgements, appended in batches.                                                                                  |
| `metrics_with_judges.csv`  | Final flat table produced by `judge_helper merge`. Includes faith/rel/comp + reasons.                                   |

Recommended analysis pivots:

- **By category × mode**: confirm or refute the expected-winner intuitions above.
- **By `vector_similarity_weight`** (E1): find the sweet spot for `vector`
  and `local` modes; expect a U-shape with extremes worse than the middle.
- **By `top_k`** (E2): plateau point for Recall@k; that's the budget-optimal k.
- **By `qa_id`**: spot questions where every mode fails — typically a dataset
  labeling error or a content gap.

---

## References

- Phase 0 commit `519bf25` — `rag ask --eval-trace / --param key=value`.
- `EvalTrace` TypedDict — [`src/ragkit/eval_context.py`](../src/ragkit/eval_context.py).
- Known parameter keys (the legal LHS of `--param key=value`) — see
  `KNOWN_PARAMS` in the same file.
