"""Evaluation instrumentation layer — context-bound parameter overrides + trace assembly.

This module provides a contextvars-backed store of tunable parameter overrides
and helpers for assembling structured evaluation traces.

Design:
  - A single ``ContextVar[dict]`` holds the per-invocation overrides.
  - Consumers replace bare constant reads with
    ``eval_context.get("<key>", original_default)``.
  - When no overrides are set, ``get()`` collapses to a single ``dict.get()``
    on the (empty) ContextVar value — zero allocation, no module-level reach.

The trace assembly helper ``build_trace(...)`` produces a fully-populated
``EvalTrace`` TypedDict from the runtime data CLI commands already collect.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
from typing import Literal, TypedDict


# --------------------------------------------------------------------------
# TypedDicts — trace schema
# --------------------------------------------------------------------------


KindT = Literal["chunk", "entity", "community", "relation", "point"]


class RetrievedItem(TypedDict):
    chunk_id: str
    rank: int
    score: float
    kind: KindT


class TimingBreakdown(TypedDict):
    embed_ms: float
    retrieve_es_ms: float
    rerank_ms: float
    generate_ms: float
    total_ms: float


class CostBreakdown(TypedDict):
    llm_calls: int
    embedding_calls: int
    tokens_in: int
    tokens_out: int
    est_cost_usd: float


class ParamsSnapshot(TypedDict):
    vector_similarity_weight: float
    similarity_threshold: float
    chunk_token_num: int
    local_top_k_seeds: int
    local_top_k_text_units: int
    local_top_k_communities: int
    local_top_k_entities: int
    local_top_k_relations: int
    global_top_k_reports: int
    map_batch_token_budget: int
    rating_threshold: int
    default_final_top_k: int


class EvalTrace(TypedDict):
    question: str
    kb: str
    mode: str
    top_k: int
    level: int | None
    timestamp_iso: str
    retrieved: list[RetrievedItem]
    timing: TimingBreakdown
    cost: CostBreakdown
    params: ParamsSnapshot
    answer: str | None


# --------------------------------------------------------------------------
# Known parameter keys + their expected types (for safe coercion)
# --------------------------------------------------------------------------


KNOWN_PARAMS: dict[str, type] = {
    "vector_similarity_weight": float,
    "similarity_threshold": float,
    "chunk_token_num": int,
    "local_top_k_seeds": int,
    "local_top_k_text_units": int,
    "local_top_k_communities": int,
    "local_top_k_entities": int,
    "local_top_k_relations": int,
    "global_top_k_reports": int,
    "map_batch_token_budget": int,
    "rating_threshold": int,
    "default_final_top_k": int,
}


# --------------------------------------------------------------------------
# ContextVar store
# --------------------------------------------------------------------------


# Default to an empty dict — every read in the no-override path is a single
# dict.get() call against this empty dict. No allocation, no work.
_OVERRIDES: contextvars.ContextVar[dict[str, object]] = contextvars.ContextVar(
    "ragkit_eval_overrides", default={}
)


def set_overrides(raw: list[str]) -> None:
    """Parse ``["key=value", ...]`` and install as the current context's overrides.

    Args:
        raw: List of ``key=value`` strings, typically from ``--param`` CLI flags.

    Raises:
        ValueError: malformed entry (no ``=``), unknown key, or failed type
            coercion.

    Notes:
        Installs a NEW dict; never mutates the existing one. Callers in a
        different contextvars context see no override (essential for tests).
    """
    parsed: dict[str, object] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(
                f"Bad --param syntax: {entry!r} — expected key=value"
            )
        key, _, value = entry.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in KNOWN_PARAMS:
            known = ", ".join(sorted(KNOWN_PARAMS))
            raise ValueError(
                f"Unknown --param key: {key!r}. Known keys: {known}"
            )
        coerce = KNOWN_PARAMS[key]
        try:
            parsed[key] = coerce(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot coerce {key}={value!r} to {coerce.__name__}: {e}"
            ) from e
    _OVERRIDES.set(parsed)


def get(name: str, default):  # noqa: ANN001 — generic passthrough
    """Return the override for ``name`` if set, else ``default``.

    Hot path: in the no-override common case this is a single dict.get() on
    an empty dict — no branch, no allocation.
    """
    return _OVERRIDES.get().get(name, default)


def current_overrides() -> dict[str, object]:
    """Read-only snapshot of the active overrides dict.

    Returns a defensive copy so callers can't mutate the live ContextVar value.
    """
    return dict(_OVERRIDES.get())


# --------------------------------------------------------------------------
# Trace assembly
# --------------------------------------------------------------------------


def build_trace(
    *,
    question: str,
    kb: str,
    mode: str,
    top_k: int,
    level: int | None,
    retrieved: list[RetrievedItem],
    timing: TimingBreakdown,
    cost: CostBreakdown,
    answer: str | None,
    defaults: ParamsSnapshot,
) -> EvalTrace:
    """Assemble a complete ``EvalTrace`` from the per-call runtime data.

    The ``params`` field merges ``defaults`` with whatever overrides the
    current context has installed — so the trace always reflects what the
    pipeline ACTUALLY ran with, not just what the defaults were.
    """
    # Start from defaults, layer overrides on top. dict() makes a shallow copy
    # so we don't mutate the caller's defaults dict.
    params: dict[str, object] = dict(defaults)
    params.update(current_overrides())

    return EvalTrace(
        question=question,
        kb=kb,
        mode=mode,
        top_k=top_k,
        level=level,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
        retrieved=retrieved,
        timing=timing,
        cost=cost,
        params=params,  # type: ignore[typeddict-item]
        answer=answer,
    )
