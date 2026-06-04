"""LLM-based consolidation of accumulated entity/relation descriptions.

When the same entity (or relation) appears in many chunks, ragkit's default
merge() just concatenates descriptions with " ". Over time these get long
and repetitive. This module re-summarizes them via an LLM.

To avoid "summary of summary" loops we use a threshold buffer:
- Triggered when description > CONSOLIDATION_THRESHOLD_CHARS (250)
- LLM is asked to produce at most CONSOLIDATION_TARGET_CHARS (180)
- The 70-char gap means a well-behaved LLM output won't re-trigger next build

If the LLM occasionally overshoots, task #24's desc_hash check will detect
the actual content change and re-embed only what truly differs.

To swap LLM providers, edit _client(). To tune cost, change max_calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from ragkit.config import get_config
from ragkit.core.graph.store import GraphStore
from ragkit.core.graph.types import Entity, Relation
from ragkit.logger import logger


# Trigger and target lengths (in characters of the description string).
# The buffer (THRESHOLD - TARGET) must be wide enough to absorb LLM overshoot;
# 70 chars is a comfortable margin for Chinese summary output.
CONSOLIDATION_THRESHOLD_CHARS = 250
CONSOLIDATION_TARGET_CHARS = 180

# A description with fewer source chunks than this likely doesn't need
# consolidation (one or two mentions = description is already focused).
CONSOLIDATION_MIN_CHUNKS = 3


ENTITY_PROMPT = """\
你是一个知识图谱编辑器。下面是从多个来源累积的关于同一实体的描述片段，请合并成一段不超过 180 字的统一描述。

【要求】
1. 保留所有关键事实：数字、日期、人名、地名、关系
2. 去除语义重复的内容（即使措辞不同）
3. 如果出现矛盾，标注"存在分歧"并保留两种说法
4. 不要添加来源里没有的信息
5. 用通顺自然的中文表达，不要罗列

【实体名】{name}
【实体类型】{type}
【累积的描述】
{description}

只返回合并后的描述文本，不要任何前缀、引号、markdown 或解释。
"""

RELATION_PROMPT = """\
你是一个知识图谱编辑器。下面是关于同一对实体之间关系的多个描述片段，请合并成一段不超过 180 字的统一描述。

【要求】
1. 保留所有关键事实
2. 去除语义重复
3. 矛盾内容标注"存在分歧"
4. 不要编造信息
5. 用通顺自然的中文表达

【实体 A】{source}
【实体 B】{target}
【累积的关系描述】
{description}

只返回合并后的描述文本，不要任何前缀、引号、markdown 或解释。
"""


@dataclass
class ConsolidationResult:
    """Outcome of one consolidate_all() invocation.

    entities_processed and relations_processed are returned for future use
    by task #24 (currently it auto-detects changes via desc_hash, but the
    sets are useful for logging and could become an optimization channel).
    """

    entities_processed: set[str] = field(default_factory=set)
    relations_processed: set[tuple[str, str]] = field(default_factory=set)
    total_calls: int = 0
    failures: int = 0


# --------------------------------------------------------------------------
# LLM client (swap point — change this to use a different provider)
# --------------------------------------------------------------------------


def _client() -> OpenAI:
    cfg = get_config()
    cfg.require_api_key()
    return OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url)


# --------------------------------------------------------------------------
# Per-item consolidation
# --------------------------------------------------------------------------


def consolidate_entity_description(entity: Entity, *, model: str | None = None) -> str | None:
    """Ask LLM to produce a consolidated description for one entity.

    Returns the new description, or None on LLM failure.
    """
    cfg = get_config()
    prompt = ENTITY_PROMPT.format(
        name=entity.name,
        type=entity.type or "unknown",
        description=entity.description,
    )
    try:
        completion = _client().chat.completions.create(
            model=model or cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            timeout=60,
        )
    except Exception as e:
        logger.warning(f"Entity '{entity.name}' consolidation LLM call failed: {e}")
        return None

    if not completion.choices:
        return None
    text = (completion.choices[0].message.content or "").strip()
    return text or None


def consolidate_relation_description(
    relation: Relation, *, model: str | None = None
) -> str | None:
    """Ask LLM to produce a consolidated description for one relation."""
    cfg = get_config()
    prompt = RELATION_PROMPT.format(
        source=relation.source,
        target=relation.target,
        description=relation.description,
    )
    try:
        completion = _client().chat.completions.create(
            model=model or cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            timeout=60,
        )
    except Exception as e:
        logger.warning(
            f"Relation {relation.source}↔{relation.target} consolidation failed: {e}"
        )
        return None

    if not completion.choices:
        return None
    text = (completion.choices[0].message.content or "").strip()
    return text or None


# --------------------------------------------------------------------------
# Batch over the whole graph
# --------------------------------------------------------------------------


def _entity_needs_consolidation(entity: Entity) -> bool:
    return (
        len(entity.description) > CONSOLIDATION_THRESHOLD_CHARS
        and len(entity.source_chunks) > CONSOLIDATION_MIN_CHUNKS
    )


def _relation_needs_consolidation(relation: Relation) -> bool:
    return (
        len(relation.description) > CONSOLIDATION_THRESHOLD_CHARS
        and len(relation.source_chunks) > CONSOLIDATION_MIN_CHUNKS
    )


def consolidate_all(
    store: GraphStore,
    *,
    max_calls: int = 20,
    progress_cb=None,
) -> ConsolidationResult:
    """Sweep the graph and consolidate descriptions over threshold.

    Args:
        store: graph backend
        max_calls: total LLM call cap across both entities and relations
        progress_cb: optional callable(stage, current, total)

    Returns:
        ConsolidationResult with sets of processed items and counts.
    """
    result = ConsolidationResult()

    # Snapshot candidates first so we don't iterate while mutating.
    entity_candidates = [e for e in store.all_entities() if _entity_needs_consolidation(e)]
    relation_candidates = [
        r for r in store.all_relations() if _relation_needs_consolidation(r)
    ]
    total_candidates = len(entity_candidates) + len(relation_candidates)
    logger.info(
        f"Consolidation: {len(entity_candidates)} entities + "
        f"{len(relation_candidates)} relations exceed threshold "
        f"(cap={max_calls})"
    )

    if total_candidates == 0:
        return result

    # Sort largest-first so we burn budget on the worst offenders.
    entity_candidates.sort(key=lambda e: len(e.description), reverse=True)
    relation_candidates.sort(key=lambda r: len(r.description), reverse=True)

    # Process entities first.
    for entity in entity_candidates:
        if result.total_calls >= max_calls:
            logger.info(f"Consolidation hit max_calls={max_calls}, stopping")
            break

        result.total_calls += 1
        if progress_cb:
            progress_cb("consolidating", result.total_calls, min(total_candidates, max_calls))

        new_desc = consolidate_entity_description(entity)
        if new_desc is None:
            result.failures += 1
            continue

        store.replace_entity_description(entity.name, new_desc)
        result.entities_processed.add(entity.name)

    # Then relations (same budget pool).
    for relation in relation_candidates:
        if result.total_calls >= max_calls:
            break

        result.total_calls += 1
        if progress_cb:
            progress_cb("consolidating", result.total_calls, min(total_candidates, max_calls))

        new_desc = consolidate_relation_description(relation)
        if new_desc is None:
            result.failures += 1
            continue

        store.replace_relation_description(relation.source, relation.target, new_desc)
        result.relations_processed.add((relation.source, relation.target))

    logger.info(
        f"Consolidation done: {len(result.entities_processed)} entities + "
        f"{len(result.relations_processed)} relations consolidated, "
        f"{result.failures} failures"
    )
    return result
