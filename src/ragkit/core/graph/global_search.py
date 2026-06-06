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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ragkit.config import get_config
from ragkit.logger import logger


# Tunables (kept as module constants so tests can patch them).
MAP_BATCH_TOKEN_BUDGET = 2000      # max tokens of report text per batch
RATING_THRESHOLD = 50              # discard points with rating < this (0-100 scale)
DEFAULT_FINAL_TOP_K = 20           # max rated points sent to the answerer
SHUFFLE_SEED = 42                  # deterministic for testing
MAX_CONCURRENT_MAP_CALLS = 5       # bounded ThreadPoolExecutor workers for Map
                                   # (respects DashScope ~1-2 QPS rate limit)


MAP_PROMPT = """\
你是知识图谱分析助手。下面是从知识库中检索到的【部分】社区报告片段。
请从这些片段中提取与用户问题相关的具体要点，并对每条要点的相关度评分。

【用户问题】
{question}

【社区报告片段】（注意：这只是相关报告的一个子集，不是全部内容）
{report_batch}

【输出格式严格 JSON】
{{
  "points": [
    {{
      "point": "一条具体的要点回答（带事实和数据）",
      "rating": 0-100 的整数,
      "source": "支撑这条要点的社区编号或来源（可选）"
    }}
  ]
}}

【评分标准 —— 务必严格遵守，采用 0-100 制】
- rating 90-100: 要点本身就是问题的明确答案
  例如：明确的数字结论、政策表态、直接结论、评级、决策。
  即使原文以事实陈述形式呈现，也应给 90-100 分。不要因为"看起来像陈述"而压低分数。
- rating 60-89: 直接相关的支撑性事实
  例如：业务方向、市场布局、关键财务数据、客户名单、技术细节。
- rating 30-59: 弱相关或推论性、间接背景信息。
- rating 10-29: 相关度极低但有微弱关联。
- rating 0-9: 完全无关或几乎无关。

【细致区分 —— 避免 LLM 齐分化】
- 避免给多条 points 完全相同的评分。即使两条都直接回答问题，
  请仔细比较哪条更准确、更具体、更直接，用细微的分差（如 92 vs 87）表达强弱。
- 如果实在判断不出差异，宁愿稍微调低其中一条（如 95 → 93），
  也不要给两条完全相同的分数 —— 后续 top-K 选择需要细粒度区分。

【硬性禁止 —— 违反将导致最终答案错误】
- 严禁产出"研报里没有 X"、"未找到 Y"、"本次未提供 Z"等否定性 points。
  你只看到全部候选社区的一个子集（batch），无权对整个研报作整体否定结论。
- 如果当前 batch 中完全没有相关信息，请直接返回 {{"points": []}}，
  不要写"本批次未发现..."这种否定性 point。

【其他要求】
1. 只基于给定的社区报告，不要编造
2. 每条 point 要具体、可引用，不要泛泛而谈
3. 全部中文输出

只返回 JSON，不要任何前缀、解释、markdown 围栏。
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
    return OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url, timeout=cfg.llm_timeout)


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

    # ISS-007: LLM occasionally returns "points": None / string / dict instead
    # of a list. `.get("points", [])` returns the WRONG-typed value (the default
    # only kicks in when the key is missing). Coerce defensively.
    raw_points = data.get("points", [])
    if not isinstance(raw_points, list):
        logger.warning(f"Global map response 'points' was not a list: {type(raw_points).__name__}")
        return []

    out: list[RatedPoint] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue  # ISS-007: skip non-dict items (e.g., list-of-strings)
        point = str(item.get("point", "")).strip()
        if not point:
            continue
        # Coerce rating safely
        try:
            rating = int(round(float(item.get("rating", 0))))
        except (TypeError, ValueError):
            rating = 0
        rating = max(0, min(100, rating))
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

    # 3) MAP: per-batch LLM scoring — concurrent with bounded workers.
    #    Each batch is independent (no shared state), so straight
    #    ThreadPoolExecutor + as_completed is safe. Order of trace output
    #    is "completion order" instead of "batch index", which is fine
    #    because final reduce is rating-sorted anyway.
    all_points: list[RatedPoint] = []
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENT_MAP_CALLS, len(batches))
    ) as executor:
        futures = {
            executor.submit(_map_rate_batch, question, batch): (i, batch)
            for i, batch in enumerate(batches, start=1)
        }
        for future in as_completed(futures):
            i, batch = futures[future]
            try:
                rir = future.result()
            except Exception as e:
                # Per-batch failure must NOT abort the whole search —
                # other batches may still contribute.
                logger.warning(f"Global map batch {i} raised: {e}")
                rir = []
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
