"""LLM answer generation, streamed as plain Python events.

No HTTP framing. The CLI consumes events directly and renders them with rich.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI

from ragkit.config import get_config
from ragkit.core.retriever import RetrievedChunk
from ragkit.logger import logger


PROMPT_TEMPLATE = """\
你是一个专业的智能助手，擅长基于提供的参考资料回答用户问题。请遵循以下原则：

**回答要求：**
1. 优先基于参考内容回答，确保答案准确可靠
2. 在回答中，每一块内容都必须标注引用的来源，格式为：##引用编号$$。例如：##1$$ 表示引用自第1条参考内容。
3. 如果参考内容不足以完全回答问题，可以结合常识补充，但需明确区分
4. 回答要条理清晰、语言自然流畅
5. 如果没有相关参考内容，请诚实说明并提供一般性建议

**参考内容：**
{references}

**用户问题：**
{question}

请基于以上信息提供专业、准确的回答。如果没有参考内容，请拒绝回答。
"""


@dataclass(frozen=True)
class Event:
    """One event emitted from the LLM stream."""

    type: str  # "content" | "thinking" | "done" | "error"
    text: str = ""
    references: tuple[RetrievedChunk, ...] = ()


def _format_references(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "暂无相关参考内容"
    lines = [f"[{c.rank}] {c.content}" for c in chunks]
    return "**知识库内容：**\n" + "\n".join(lines)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build the final prompt sent to the LLM. Public for testability."""
    return PROMPT_TEMPLATE.format(
        question=question,
        references=_format_references(chunks),
    )


def generate(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    enable_thinking: bool = True,
) -> Iterator[Event]:
    """Stream answer events for `question` using `chunks` as context.

    Always yields one final Event with type='done' carrying references.
    On error, yields type='error' and terminates.
    """
    cfg = get_config()
    cfg.require_api_key()

    prompt = build_prompt(question, chunks)
    logger.debug(f"Prompt length: {len(prompt)} chars")

    client = OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url, timeout=cfg.llm_timeout)

    try:
        stream = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            extra_body={"enable_thinking": enable_thinking} if enable_thinking else None,
        )

        for piece in stream:
            choice = piece.choices[0]
            if choice.finish_reason == "stop":
                break

            delta = choice.delta
            content = getattr(delta, "content", None)
            thinking = getattr(delta, "reasoning_content", None)

            if content:
                yield Event(type="content", text=content)
            elif thinking:
                yield Event(type="thinking", text=thinking)

        yield Event(type="done", references=tuple(chunks))

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        yield Event(type="error", text=str(e))
