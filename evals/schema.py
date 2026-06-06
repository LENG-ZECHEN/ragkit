"""Dataset, judge, and sweep-result schemas for the ragkit eval harness.

All types here are stdlib-only (``dataclasses`` + ``typing``). Phase 2 scripts
import these symbols to read the dataset, type judge output, and write sweep
result rows; no runtime logic lives in this module beyond two JSONL helpers.

Design notes:
  - ``QAItem`` is ``frozen=True`` — dataset rows are immutable values.
  - ``JudgeScores`` is a ``TypedDict`` so a JSON-decoded dict can be cast to it
    without copy.
  - ``SweepResultRow`` embeds the raw ``EvalTrace`` (re-exported from
    ``ragkit.eval_context``) plus derived retrieval metrics + optional judge
    scores, so one row is a self-contained record of a single retrieval run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

# Re-export the trace TypedDict so eval code has a single import surface.
from ragkit.eval_context import EvalTrace  # noqa: F401  (re-exported)


# --------------------------------------------------------------------------
# Dataset types
# --------------------------------------------------------------------------


QACategory = Literal[
    "factual",
    "passage_quoted",
    "cross_paragraph_theme",
    "refusal",
]


@dataclass(frozen=True)
class QAItem:
    """One labeled QA pair in the eval dataset.

    Attributes:
        id: Stable identifier (e.g. ``"fact-001"``).
        question: User-facing question, Chinese.
        category: Which evaluation bucket this question tests.
        ground_truth_chunk_ids: Chunk IDs that SHOULD appear in retrieval. Used
            to compute Recall@k, MRR, nDCG@10. Empty list = refusal-case (no
            chunk should match).
        gold_answer: Reference answer text, or ``None`` for refusal-case.
        notes: Free-text annotator notes; ignored by metrics.
    """

    id: str
    question: str
    category: QACategory
    ground_truth_chunk_ids: list[str] = field(default_factory=list)
    gold_answer: str | None = None
    notes: str | None = None


# --------------------------------------------------------------------------
# Judge output
# --------------------------------------------------------------------------


class JudgeScores(TypedDict):
    """Structured output expected from the LLM judge.

    Each dimension is an integer score in ``[1, 5]`` plus a short reason. See
    ``evals/judge_prompts.py`` for the rubric anchors.
    """

    faithfulness: int
    faithfulness_reason: str
    relevance: int
    relevance_reason: str
    completeness: int
    completeness_reason: str


# --------------------------------------------------------------------------
# Sweep result row
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResultRow:
    """One row of a sweep — one (qa_item × mode × param-combo) outcome.

    Attributes:
        qa_id: ID of the ``QAItem`` this row evaluates.
        mode: Retrieval mode (``"vector"`` | ``"local"`` | ``"global"``).
        params: The exact override dict in effect for this run.
        trace: Full ``EvalTrace`` as emitted by ``--eval-trace`` (``None`` on
            subprocess failure).
        recall_at_k: Recall computed at each k in ``{1, 3, 5, 10}``.
        mrr: Mean Reciprocal Rank for this single query.
        ndcg_at_10: Normalized DCG over the top-10 retrieved items.
        refusal_correct: True iff this is a refusal-case row AND retrieval
            correctly returned no chunks. False otherwise (incl. non-refusal).
        retrieved_contents: ``[{chunk_id, content}]`` fetched from ES, one per
            chunk-kind retrieved item, in retrieval order. Empty list on
            failure or for non-chunk items.

    P2.5 change: ``judge`` field removed — judging is now human-in-the-loop
    via ``evals/judge_helper.py`` and merged into the final metrics.csv
    downstream of this sweep.
    """

    qa_id: str
    mode: str
    params: dict[str, object]
    trace: dict | None  # EvalTrace as a dict — JSON-roundtrip safe.
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_10: float
    refusal_correct: bool = False
    retrieved_contents: list[dict[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Dataset JSONL helpers
# --------------------------------------------------------------------------


def load_dataset(path: Path) -> list[QAItem]:
    """Read a dataset JSONL file into a list of immutable ``QAItem``s.

    Each line must decode to a JSON object with at least ``id``, ``question``,
    and ``category``; optional fields fall back to their dataclass defaults.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a row is missing a required field.
    """
    items: list[QAItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            try:
                items.append(
                    QAItem(
                        id=obj["id"],
                        question=obj["question"],
                        category=obj["category"],
                        ground_truth_chunk_ids=list(
                            obj.get("ground_truth_chunk_ids", [])
                        ),
                        gold_answer=obj.get("gold_answer"),
                        notes=obj.get("notes"),
                    )
                )
            except KeyError as e:
                raise ValueError(
                    f"{path}:{lineno}: missing required field {e!s}"
                ) from e
    return items


def save_dataset(items: list[QAItem], path: Path) -> None:
    """Write ``items`` to ``path`` as JSONL (one object per line).

    The output is deterministic: keys sorted, no trailing whitespace, and the
    file always ends with a single newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            line = json.dumps(asdict(item), ensure_ascii=False, sort_keys=True)
            fh.write(line + "\n")
