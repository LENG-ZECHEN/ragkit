"""LLM judge prompts for the ragkit eval harness (Chinese).

The judge scores a generated answer along three dimensions on a 1-5 scale:

  * Faithfulness — is every claim grounded in the retrieved context?
  * Relevance    — does the answer actually address the question?
  * Completeness — does it cover the key facts the gold answer covers?

The judge MUST return strict JSON (no prose, no code fence). Phase 2 parses the
output into ``JudgeScores`` (see ``evals/schema.py``).
"""

from __future__ import annotations

from .schema import QAItem


# --------------------------------------------------------------------------
# System prompt — role, behaviour, output contract
# --------------------------------------------------------------------------


SYSTEM_PROMPT_CN = """你是一名严格的中文 RAG 答案评审。你的任务是依据用户问题、参考答案和系统检索到的上下文,对系统生成的答案在三个维度上打分:

1. 忠实度 (faithfulness): 答案中的每一项断言是否都能从“检索到的上下文”中找到依据。
2. 相关性 (relevance): 答案是否切题、直接回应了用户的问题。
3. 完整性 (completeness): 答案是否覆盖了“参考答案”中所有关键信息点。

打分规则:
- 每个维度均为 1-5 的整数,1 最差、5 最好。
- 不允许给出中间分数(如 3.5)。
- 若“参考答案”为 null,说明这是一道“应拒答”问题:
  * 系统正确拒答或明确表示资料中没有相关信息 → 三项均给 5;
  * 系统强行编造答案 → faithfulness 给 1。

输出要求(极其重要):
- 只输出一个严格合法的 JSON 对象,不要包含任何额外文字、解释或 Markdown 代码块。
- JSON 必须包含且只包含以下 6 个键:
  faithfulness, faithfulness_reason,
  relevance, relevance_reason,
  completeness, completeness_reason
- 三个 *_reason 字段为不超过 60 字的中文简短理由。
"""


# --------------------------------------------------------------------------
# User prompt template
# --------------------------------------------------------------------------


USER_PROMPT_TEMPLATE_CN = """【用户问题】
{question}

【参考答案】
{gold_answer}

【系统检索到的上下文】
{retrieved_context}

【系统生成的答案】
{generated_answer}

【打分锚点(用于校准)】

忠实度 faithfulness:
  5 — 所有断言均可在上下文中逐句对齐;
  3 — 主要断言有据,但存在少量未在上下文中出现的细节;
  1 — 明显与上下文矛盾,或大量内容凭空编造。

相关性 relevance:
  5 — 直接、准确地回应了问题,不跑题;
  3 — 部分回应,夹杂无关信息;
  1 — 答非所问。

完整性 completeness:
  5 — 覆盖参考答案的全部关键信息点;
  3 — 覆盖约一半;
  1 — 关键信息几乎全部缺失。

请严格按 system 指令输出 JSON。
"""


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def build_judge_prompt(
    qa: QAItem,
    retrieved_context: str,
    generated_answer: str,
) -> tuple[str, str]:
    """Return ``(system, user)`` strings ready for the judge LLM call.

    Args:
        qa: The dataset item being evaluated.
        retrieved_context: Concatenated text of retrieved chunks (caller
            chooses the join strategy — typically chunk-separator newlines).
        generated_answer: The system's answer for this question.

    Notes:
        ``gold_answer`` of ``None`` is rendered as the literal ``"null"`` so
        the judge can apply the refusal-case rule defined in the system prompt.
    """
    gold = qa.gold_answer if qa.gold_answer is not None else "null"
    user = USER_PROMPT_TEMPLATE_CN.format(
        question=qa.question,
        gold_answer=gold,
        retrieved_context=retrieved_context,
        generated_answer=generated_answer,
    )
    return SYSTEM_PROMPT_CN, user
