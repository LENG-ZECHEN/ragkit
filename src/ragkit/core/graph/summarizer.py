"""LLM-based structured report for each community.

Inspired by Microsoft GraphRAG's community report format. Each community
gets a structured report with:

- title             (short, ≤20 chars)
- summary           (paragraph, 100-200 chars)
- rank              (1-10 importance score)
- rank_explanation  (one-sentence rationale)
- findings          (list of key facts, each with summary + explanation)

The structured format gives the retriever multiple high-value passages
per community (vs one blob string), and provides a richer prompt context
than a flat summary.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from ragkit.config import get_config
from ragkit.core.graph.store import GraphStore
from ragkit.core.graph.types import Community, Finding
from ragkit.logger import logger


MAX_FINDINGS_PER_COMMUNITY = 5


SUMMARY_PROMPT = """\
你是知识图谱社区分析助手。基于下面给定的实体和关系，对这个社区生成一份结构化报告。

【实体列表】
{entities}

【关系列表】
{relations}

【输出格式严格 JSON】
{{
  "title": "简短标题，10-20 字",
  "summary": "段落摘要，100-200 字，描述这个群组的主题、主要实体、关键关系",
  "rank": 1-10 的整数（重要性，根据实体数量+关系密度+实体重要性综合判断）,
  "rank_explanation": "为什么打这个分数（一句话）",
  "findings": [
    {{
      "summary": "一行关键结论",
      "explanation": "详细说明 50-100 字"
    }}
  ]
}}

【要求】
1. title 简短能概括主题
2. summary 必须是连贯的描述，不要罗列实体
3. findings 选 3-5 条最关键的事实/洞察，按重要性排序，最多 5 条
4. 不要编造，只基于给定的实体和关系
5. 全部中文

只返回 JSON，不要任何前缀、引号、markdown 围栏。
"""


def _client() -> OpenAI:
    cfg = get_config()
    cfg.require_api_key()
    return OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url)


def _strip_code_fence(text: str) -> str:
    """Defensively strip ```json``` fences if the LLM ignored instructions."""
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _build_community_context(community: Community, store: GraphStore) -> tuple[str, str]:
    """Render entity bullets + relation bullets for one community's prompt."""
    entity_lines: list[str] = []
    for name in community.entity_names:
        e = store.get_entity(name)
        if e:
            desc = e.description or "（无描述）"
            entity_lines.append(f"- {e.name} [{e.type}]: {desc}")

    member_set = set(community.entity_names)
    relation_lines: list[str] = []
    for r in store.all_relations():
        if r.source in member_set and r.target in member_set:
            relation_lines.append(
                f"- {r.source} → {r.target}: {r.description or '（无描述）'}"
            )

    return "\n".join(entity_lines), "\n".join(relation_lines) or "（无内部关系）"


def _parse_report(raw: str, community: Community) -> dict:
    """Parse the LLM's JSON output into the fields we need.

    Tolerant of code fences, missing fields, and over-long findings lists.
    Returns a dict with the structured fields; consumer applies them to
    the Community object.

    On parse failure: returns a minimal dict so the community still has
    a usable (if degraded) report.
    """
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Community {community.id} report JSON parse failed: {e}")
        return {
            "title": f"Community {community.id}",
            "summary": "",
            "rank": 0.0,
            "rank_explanation": "",
            "findings": [],
        }

    title = str(data.get("title", f"Community {community.id}")).strip()
    summary = str(data.get("summary", "")).strip()

    # rank may be int, float, or string — coerce safely
    raw_rank = data.get("rank", 0)
    try:
        rank = float(raw_rank)
    except (TypeError, ValueError):
        rank = 0.0
    # Clamp to advertised range
    rank = max(0.0, min(10.0, rank))

    rank_explanation = str(data.get("rank_explanation", "")).strip()

    findings: list[dict] = []
    for raw_finding in data.get("findings", [])[:MAX_FINDINGS_PER_COMMUNITY]:
        f_summary = str(raw_finding.get("summary", "")).strip()
        f_explanation = str(raw_finding.get("explanation", "")).strip()
        if f_summary or f_explanation:
            findings.append({"summary": f_summary, "explanation": f_explanation})

    return {
        "title": title,
        "summary": summary,
        "rank": rank,
        "rank_explanation": rank_explanation,
        "findings": findings,
    }


def generate_community_report(
    community: Community, store: GraphStore, *, model: str | None = None
) -> None:
    """Fill community.title / summary / rank / rank_explanation / findings
    in place. Falls back to safe defaults on LLM failure (no exception)."""
    if not community.entity_names:
        return

    cfg = get_config()
    entities_text, relations_text = _build_community_context(community, store)
    prompt = SUMMARY_PROMPT.format(entities=entities_text, relations=relations_text)

    try:
        completion = _client().chat.completions.create(
            model=model or cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
            timeout=60,
        )
    except Exception as e:
        logger.warning(f"Community {community.id} report LLM call failed: {e}")
        return

    if not completion.choices:
        return

    raw = completion.choices[0].message.content or ""
    report = _parse_report(raw, community)

    community.title = report["title"]
    community.summary = report["summary"]
    community.rank = report["rank"]
    community.rank_explanation = report["rank_explanation"]
    community.findings = [
        Finding(summary=f["summary"], explanation=f["explanation"])
        for f in report["findings"]
    ]


def summarize_community(
    community: Community, store: GraphStore, *, model: str | None = None
) -> str:
    """Legacy single-string summary entry point.

    Now implemented as: generate the structured report, then return only
    the summary field. Kept for any caller still using the old name.
    """
    generate_community_report(community, store, model=model)
    return community.summary


def summarize_all(
    store: GraphStore,
    *,
    max_communities: int | None = None,
    progress_cb=None,
) -> int:
    """Generate a structured report for each community in the store.

    Important: mutates community objects in place; does NOT replace the
    list. ``max_communities`` controls how many get processed — any beyond
    that limit retain their existing (possibly empty) reports rather than
    being silently dropped.

    Returns the number of communities for which LLM-generated reports
    were empty (failure or no output).
    """
    communities = store.all_communities()
    targets = communities if max_communities is None else communities[:max_communities]

    failures = 0
    for i, c in enumerate(targets, start=1):
        generate_community_report(c, store)
        if not c.summary and not c.title and not c.findings:
            failures += 1
        if progress_cb:
            progress_cb("summarizing", i, len(targets))
        logger.debug(
            f"Community {c.id} (L{c.level}): "
            f"title={c.title!r}, summary={len(c.summary)} chars, "
            f"{len(c.findings)} findings"
        )

    store.set_communities(communities)
    return failures
