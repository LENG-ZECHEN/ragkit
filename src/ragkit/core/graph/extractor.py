"""Extract entities and relations from a chunk using an LLM.

To swap the extraction model, change EXTRACTION_MODEL or the prompt.
To swap the LLM provider entirely, replace `_llm_client()` body.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from ragkit.config import get_config
from ragkit.core.graph.types import Entity, Relation
from ragkit.logger import logger

# Microsoft-GraphRAG-style prompt, simplified for production reliability.
EXTRACTION_PROMPT = """\
你是一个知识图谱抽取器。从下面给定的文本中抽取出关键实体和它们之间的关系。

【实体类型可选】person（人物）、organization（机构/公司）、location（地点）、\
concept（概念/技术/产品）、event（事件）、metric（指标/数据）、other（其他）

【输出格式严格 JSON】
{{
  "entities": [
    {{"name": "实体名（统一小写、去掉空格）", "type": "类型", "description": "一句话描述该实体在文本中的角色"}}
  ],
  "relations": [
    {{"source": "实体A", "target": "实体B", "description": "他们之间的关系"}}
  ]
}}

【要求】
1. 只抽取在文本中真实出现且有明确含义的实体
2. relations 中的 source/target 必须是 entities 列表中已经出现的 name
3. 不要编造，不要在 description 里加自己的猜测
4. 如果文本太短或没有实体，返回 {{"entities": [], "relations": []}}

【文本】
{text}

只返回 JSON，不要包含任何其他文字、markdown 围栏。
"""


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]


def _llm_client() -> OpenAI:
    cfg = get_config()
    cfg.require_api_key()
    return OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url)


def _strip_code_fence(text: str) -> str:
    """LLMs occasionally wrap JSON in ```json``` fences despite instructions."""
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _parse_extraction(raw: str, chunk_id: str) -> ExtractionResult:
    """Parse the LLM JSON output. Tolerant of code fences and minor noise."""
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse extraction JSON for chunk {chunk_id}: {e}")
        return ExtractionResult(entities=[], relations=[])

    entities: list[Entity] = []
    seen_names: set[str] = set()
    for e in data.get("entities", []):
        name = str(e.get("name", "")).strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        entities.append(Entity(
            name=name,
            type=str(e.get("type", "unknown")).strip() or "unknown",
            description=str(e.get("description", "")).strip(),
            source_chunks=[chunk_id],
        ))

    relations: list[Relation] = []
    # Only allow edges whose endpoints are in the entity set we just kept,
    # to avoid dangling references the LLM sometimes hallucinates.
    valid = {e.name.lower() for e in entities}
    for r in data.get("relations", []):
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        if not src or not tgt or src.lower() == tgt.lower():
            continue
        if src.lower() not in valid or tgt.lower() not in valid:
            continue
        relations.append(Relation(
            source=src,
            target=tgt,
            description=str(r.get("description", "")).strip(),
            weight=1.0,
            source_chunks=[chunk_id],
        ))
    return ExtractionResult(entities=entities, relations=relations)


def extract_from_text(text: str, chunk_id: str, *, model: str | None = None) -> ExtractionResult:
    """Run LLM extraction on a single chunk."""
    if not text or not text.strip():
        return ExtractionResult(entities=[], relations=[])

    cfg = get_config()
    use_model = model or cfg.llm_model

    client = _llm_client()
    try:
        completion = client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
            response_format={"type": "json_object"},
            stream=False,
            timeout=60,
        )
    except Exception as e:
        logger.error(f"Extraction LLM call failed for chunk {chunk_id}: {e}")
        return ExtractionResult(entities=[], relations=[])

    if not completion.choices:
        return ExtractionResult(entities=[], relations=[])
    raw = completion.choices[0].message.content or ""
    return _parse_extraction(raw, chunk_id)
