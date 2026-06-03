"""LLM-based summary for each community.

Summaries are the unit of retrieval for global queries — short paragraphs
that capture what a cluster of entities is collectively "about".
"""

from __future__ import annotations

from openai import OpenAI

from ragkit.config import get_config
from ragkit.core.graph.store import GraphStore
from ragkit.core.graph.types import Community
from ragkit.logger import logger


SUMMARY_PROMPT = """\
你是一个知识图谱分析助手。基于下面给定的实体和关系，写一段简洁的摘要（中文，100-200 字）。
摘要应该说明：(1) 这个群组讨论的是什么主题；(2) 主要实体及其角色；(3) 关键关系。
不要逐条列举，要凝练成一段连贯的描述。

【实体】
{entities}

【关系】
{relations}

直接输出摘要文字，不要加任何前缀、标题或 markdown。
"""


def _client() -> OpenAI:
    cfg = get_config()
    cfg.require_api_key()
    return OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url)


def _build_community_context(community: Community, store: GraphStore) -> tuple[str, str]:
    """Render entity bullets + relation bullets for one community."""
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
            relation_lines.append(f"- {r.source} → {r.target}: {r.description or '（无描述）'}")

    return "\n".join(entity_lines), "\n".join(relation_lines) or "（无内部关系）"


def summarize_community(community: Community, store: GraphStore, *, model: str | None = None) -> str:
    """Return a summary string for one community. Empty on LLM failure."""
    if not community.entity_names:
        return ""

    cfg = get_config()
    entities_text, relations_text = _build_community_context(community, store)

    prompt = SUMMARY_PROMPT.format(entities=entities_text, relations=relations_text)
    try:
        completion = _client().chat.completions.create(
            model=model or cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            timeout=60,
        )
    except Exception as e:
        logger.warning(f"Community {community.id} summary failed: {e}")
        return ""

    if not completion.choices:
        return ""
    return (completion.choices[0].message.content or "").strip()


def summarize_all(
    store: GraphStore,
    *,
    max_communities: int | None = None,
    progress_cb=None,
) -> int:
    """Summarize the top-N largest communities in place.

    Important: mutates `Community.summary` on the existing community objects
    rather than replacing the list. Communities beyond `max_communities`
    keep their existing (possibly empty) summaries — they are NOT dropped
    from the store, which would be a data-loss bug.

    Returns the number of failed LLM calls (so the caller can decide
    whether to abort persisting).
    """
    communities = store.all_communities()
    targets = communities if max_communities is None else communities[:max_communities]

    failures = 0
    for i, c in enumerate(targets, start=1):
        summary = summarize_community(c, store)
        if not summary:
            failures += 1
        c.summary = summary
        if progress_cb:
            progress_cb("summarizing", i, len(targets))
        logger.debug(f"Community {c.id}: {len(c.summary)} chars")

    # Write the full list back (with mutated summaries on the head of the list).
    store.set_communities(communities)
    return failures
