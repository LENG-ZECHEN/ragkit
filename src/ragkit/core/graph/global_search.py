"""Global Search (Map-Reduce) — Microsoft GraphRAG style.

Architecture:

    User Query
       │
       ▼  vector search over community reports
    Top-K reports (across levels, or one specific level)
       │
       ▼  shuffle + batch by token budget
    Batches B1, B2, ... BN
       │
       ▼  MAP: each batch → LLM → list of (point, rating)
    All rated points
       │
       ▼  REDUCE: filter by rating threshold, sort, top-N
    Final aggregated points
       │
       ▼  caller (generator) composes the final answer using these as context

This module owns the Map and Reduce steps. The vector search step lives
in searcher.py; the final answer composition lives in the existing
generator.py.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ragkit.config import get_config
from ragkit.logger import logger


# Tunables (kept as module constants so tests can patch them).
MAP_BATCH_TOKEN_BUDGET = 2000      # max tokens of report text per batch
RATING_THRESHOLD = 5               # discard points with rating < this
DEFAULT_FINAL_TOP_K = 20           # max rated points sent to the answerer
SHUFFLE_SEED = 42                  # deterministic for testing


MAP_PROMPT = """\
你是知识图谱分析助手。下面是从知识库中检索到的社区报告片段。
基于这些片段，针对用户问题给出多条要点回答。

【用户问题】
{question}

【社区报告片段】
{report_batch}

【输出格式严格 JSON】
{{
  "points": [
    {{
      "point": "一条具体的要点回答（带事实和数据）",
      "rating": 1-10 的整数（这条要点对回答该问题的重要程度）,
      "source": "支撑这条要点的社区编号或来源（可选）"
    }}
  ]
}}

【要求】
1. 只基于给定的社区报告，不要编造
2. 每条 point 要具体、有信息量，不要泛泛而谈
3. rating 评分：10=直接回答问题、8=高度相关、5=部分相关、3 以下=几乎不相关
4. 如果片段里没有任何能回答问题的信息，返回 {{"points": []}}
5. 全部中文

只返回 JSON，不要任何前缀、引号、markdown 围栏。
"""


@dataclass
class RatedPoint:
    """One scored answer point produced by the Map step."""

    point: str
    rating: int
    source: str = ""


def _client() -> OpenAI:
    cfg = get_config()
    cfg.require_api_key()
    return OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url)


# --------------------------------------------------------------------------
# Helpers — shuffle + token-budget batching
# --------------------------------------------------------------------------


def _shuffle_with_seed(items: list, seed: int = SHUFFLE_SEED) -> list:
    """Deterministic shuffle so tests are stable."""
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — 1 token ≈ 2 Chinese chars or 4 English chars.

    Microsoft uses tiktoken; we keep it cheap and approximate to avoid
    yet another dep call in the hot path. ~5% off doesn't matter for
    batching decisions.
    """
    chinese = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - chinese
    return chinese // 2 + other // 4 + 1


def _format_report_for_batch(report: dict[str, Any]) -> str:
    """Render one community report's ES _source dict into prompt text."""
    cid = report.get("community_id_int", "?")
    level = report.get("community_level_int", "?")
    content = report.get("content_with_weight") or ""
    return f"[Community {cid} · L{level}]\n{content}"


def _batch_by_token_count(
    reports: list[dict[str, Any]], max_tokens: int = MAP_BATCH_TOKEN_BUDGET
) -> list[list[dict[str, Any]]]:
    """Greedy pack reports into batches that each fit under max_tokens."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for report in reports:
        rendered = _format_report_for_batch(report)
        tokens = _estimate_tokens(rendered)
        # A single oversized report still gets its own batch.
        if current and current_tokens + tokens > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(report)
        current_tokens += tokens

    if current:
        batches.append(current)
    return batches


# --------------------------------------------------------------------------
# MAP step
# --------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _parse_map_response(raw: str) -> list[RatedPoint]:
    """Parse one map-batch LLM response into RatedPoint list.

    Tolerant of code fences, missing keys, non-integer ratings.
    Returns [] on parse failure.
    """
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Global map response JSON parse failed: {e}")
        return []

    out: list[RatedPoint] = []
    for item in data.get("points", []):
        point = str(item.get("point", "")).strip()
        if not point:
            continue
        # Coerce rating safely
        try:
            rating = int(round(float(item.get("rating", 0))))
        except (TypeError, ValueError):
            rating = 0
        rating = max(0, min(10, rating))
        source = str(item.get("source", "")).strip()
        out.append(RatedPoint(point=point, rating=rating, source=source))
    return out


def _map_rate_batch(question: str, batch: list[dict[str, Any]]) -> list[RatedPoint]:
    """One LLM call: feed a batch of community reports + the question;
    receive rated points."""
    cfg = get_config()
    report_text = "\n\n".join(_format_report_for_batch(r) for r in batch)
    prompt = MAP_PROMPT.format(question=question, report_batch=report_text)

    try:
        completion = _client().chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
            timeout=60,
        )
    except Exception as e:
        logger.warning(f"Global map LLM call failed (batch size={len(batch)}): {e}")
        return []

    if not completion.choices:
        return []
    return _parse_map_response(completion.choices[0].message.content or "")


# --------------------------------------------------------------------------
# REDUCE step
# --------------------------------------------------------------------------


def _reduce_rated_points(
    all_points: list[RatedPoint],
    *,
    threshold: int = RATING_THRESHOLD,
    top_k: int = DEFAULT_FINAL_TOP_K,
) -> list[RatedPoint]:
    """Filter low-rated points and keep the top-K by rating."""
    kept = [p for p in all_points if p.rating >= threshold]
    kept.sort(key=lambda p: -p.rating)
    return kept[:top_k]


# --------------------------------------------------------------------------
# Main pipeline (callable from retriever.py)
# --------------------------------------------------------------------------


def run_global_search(
    question: str,
    community_reports: list[dict[str, Any]],
    *,
    max_tokens_per_batch: int = MAP_BATCH_TOKEN_BUDGET,
    rating_threshold: int = RATING_THRESHOLD,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
) -> list[RatedPoint]:
    """Run the full Map-Reduce pipeline on the given community reports.

    Args:
        question: user query
        community_reports: ES _source dicts for the candidate communities
            (caller does the vector search to populate this)
        max_tokens_per_batch: token budget per map call
        rating_threshold: min rating to survive reduce
        final_top_k: cap on returned points

    Returns:
        List of RatedPoint, sorted by rating desc, ready for the generator
        to compose into the final answer.
    """
    from ragkit.cli import observe

    if not community_reports:
        return []

    # 1) Deterministic shuffle to reduce ordering bias when LLM sees adjacent
    #    reports first.
    shuffled = _shuffle_with_seed(community_reports)

    # 2) Pack into batches under the token budget.
    batches = _batch_by_token_count(shuffled, max_tokens=max_tokens_per_batch)
    observe.trace_global_batches(
        [len(b) for b in batches],
        [sum(_estimate_tokens(_format_report_for_batch(r)) for r in b) for b in batches],
    )

    # 3) MAP: per-batch LLM scoring.
    all_points: list[RatedPoint] = []
    for i, batch in enumerate(batches, start=1):
        rir = _map_rate_batch(question, batch)
        observe.trace_global_map_batch(i, len(batch), rir)
        all_points.extend(rir)

    # 4) REDUCE: filter + top-K.
    final = _reduce_rated_points(
        all_points,
        threshold=rating_threshold,
        top_k=final_top_k,
    )
    observe.trace_global_reduce(len(all_points), len(final), rating_threshold)
    return final
