"""Tests for ragkit.eval_context — the contextvars-backed parameter override
store and the trace assembly helper.

These cover:
  * basic get/set with type coercion
  * malformed input rejection
  * context isolation (essential for the no-override invariant — overrides set
    inside one ``contextvars.copy_context()`` must NOT bleed into another)
  * build_trace produces a complete EvalTrace with merged params
"""

from __future__ import annotations

import contextvars

import pytest

from ragkit import eval_context


@pytest.fixture(autouse=True)
def _reset_overrides():
    """Each test starts with no overrides installed in the calling context.

    contextvars.ContextVar.set returns a Token we can use to reset, so we
    capture the current value (or default), let the test run, then restore.
    """
    # The cleanest reset is to install an empty dict explicitly. The default
    # value of an empty dict is shared across tests via the ContextVar default,
    # but set_overrides([]) is a clean way to reset to known empty state.
    eval_context.set_overrides([])
    yield
    eval_context.set_overrides([])


# ==========================================================================
# Basic get / set semantics
# ==========================================================================


def test_get_returns_default_when_no_override():
    assert eval_context.get("vector_similarity_weight", 0.6) == 0.6


def test_set_then_get_returns_override():
    eval_context.set_overrides(["vector_similarity_weight=0.5"])
    assert eval_context.get("vector_similarity_weight", 0.6) == 0.5


def test_get_ignores_unrelated_overrides():
    """Setting one override must NOT change the lookup of another key."""
    eval_context.set_overrides(["vector_similarity_weight=0.4"])
    # The default still applies for similarity_threshold.
    assert eval_context.get("similarity_threshold", 0.1) == 0.1


def test_set_overrides_coerces_int():
    eval_context.set_overrides(["chunk_token_num=256"])
    val = eval_context.get("chunk_token_num", 128)
    assert val == 256
    assert isinstance(val, int)


def test_set_overrides_coerces_float():
    eval_context.set_overrides(["vector_similarity_weight=0.75"])
    val = eval_context.get("vector_similarity_weight", 0.6)
    assert val == 0.75
    assert isinstance(val, float)


def test_set_overrides_multiple_keys():
    eval_context.set_overrides([
        "vector_similarity_weight=0.3",
        "chunk_token_num=512",
        "rating_threshold=70",
    ])
    assert eval_context.get("vector_similarity_weight", 0.6) == 0.3
    assert eval_context.get("chunk_token_num", 128) == 512
    assert eval_context.get("rating_threshold", 50) == 70


# ==========================================================================
# Error handling
# ==========================================================================


def test_set_overrides_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown --param key"):
        eval_context.set_overrides(["unknown_key=1"])


def test_set_overrides_rejects_malformed_entry_no_equals():
    with pytest.raises(ValueError, match="Bad --param syntax"):
        eval_context.set_overrides(["bad_format"])


def test_set_overrides_rejects_bad_float():
    with pytest.raises(ValueError, match="Cannot coerce"):
        eval_context.set_overrides(["vector_similarity_weight=not_a_float"])


def test_set_overrides_rejects_bad_int():
    with pytest.raises(ValueError, match="Cannot coerce"):
        eval_context.set_overrides(["chunk_token_num=not_a_number"])


def test_set_overrides_failure_is_atomic():
    """If one entry in the list is bad, NONE of them should be applied —
    the install is atomic. This avoids partial-state surprises."""
    eval_context.set_overrides(["vector_similarity_weight=0.4"])
    with pytest.raises(ValueError):
        eval_context.set_overrides([
            "chunk_token_num=200",  # valid
            "unknown=1",            # bad
        ])
    # The prior good install survives because the second call raised before
    # calling _OVERRIDES.set(). The first install's value should still be active.
    assert eval_context.get("vector_similarity_weight", 0.6) == 0.4
    # And the would-be-new value did NOT get applied.
    assert eval_context.get("chunk_token_num", 128) == 128


# ==========================================================================
# Context isolation — critical for safety in async / parallel test runs
# ==========================================================================


def test_overrides_do_not_leak_across_contexts():
    """Each contextvars context has its own _OVERRIDES dict.

    If we set overrides inside copy_context().run(), an outer context (or a
    sibling context) must NOT see those overrides.
    """
    # Outer context: install one override.
    eval_context.set_overrides(["vector_similarity_weight=0.9"])

    seen: dict[str, object] = {}

    def inside():
        # Inside this copied context, install a different override.
        eval_context.set_overrides(["vector_similarity_weight=0.1"])
        seen["inside"] = eval_context.get("vector_similarity_weight", 0.6)

    ctx = contextvars.copy_context()
    ctx.run(inside)

    # Outer context's value is unchanged: 0.9 still applies.
    assert eval_context.get("vector_similarity_weight", 0.6) == 0.9
    # The inner context saw its own override of 0.1.
    assert seen["inside"] == 0.1


def test_current_overrides_empty_by_default():
    assert eval_context.current_overrides() == {}


def test_current_overrides_returns_defensive_copy():
    """current_overrides() must return a copy so callers can't mutate state."""
    eval_context.set_overrides(["vector_similarity_weight=0.5"])
    snap = eval_context.current_overrides()
    snap["vector_similarity_weight"] = 999.0  # mutate the snapshot
    # The stored override must be unchanged.
    assert eval_context.get("vector_similarity_weight", 0.6) == 0.5


# ==========================================================================
# build_trace
# ==========================================================================


def _defaults() -> dict:
    """Reusable ParamsSnapshot of defaults."""
    return {
        "vector_similarity_weight": 0.6,
        "similarity_threshold": 0.1,
        "chunk_token_num": 128,
        "local_top_k_seeds": 10,
        "local_top_k_text_units": 5,
        "local_top_k_communities": 3,
        "local_top_k_entities": 8,
        "local_top_k_relations": 8,
        "global_top_k_reports": 30,
        "map_batch_token_budget": 2000,
        "rating_threshold": 50,
        "default_final_top_k": 20,
    }


def test_build_trace_has_all_required_keys():
    """A complete EvalTrace must carry every documented field — downstream
    consumers (eval harness, dashboard) rely on the full schema."""
    trace = eval_context.build_trace(
        question="What is X?",
        kb="kb1",
        mode="vector",
        top_k=5,
        level=None,
        retrieved=[],
        timing={
            "embed_ms": 1.0, "retrieve_es_ms": 2.0, "rerank_ms": 0.5,
            "generate_ms": 3.0, "total_ms": 6.5,
        },
        cost={
            "llm_calls": 1, "embedding_calls": 1, "tokens_in": 100,
            "tokens_out": 200, "est_cost_usd": 0.0,
        },
        answer="answer text",
        defaults=_defaults(),  # type: ignore[arg-type]
    )

    for key in (
        "question", "kb", "mode", "top_k", "level", "timestamp_iso",
        "retrieved", "timing", "cost", "params", "answer",
    ):
        assert key in trace, f"missing key: {key}"

    assert trace["question"] == "What is X?"
    assert trace["kb"] == "kb1"
    assert trace["mode"] == "vector"
    assert trace["top_k"] == 5
    assert trace["level"] is None
    assert trace["answer"] == "answer text"


def test_build_trace_params_reflect_defaults_when_no_overrides():
    """No overrides → params is exactly the defaults dict."""
    trace = eval_context.build_trace(
        question="q", kb="k", mode="vector", top_k=5, level=None,
        retrieved=[],
        timing={"embed_ms": 0, "retrieve_es_ms": 0, "rerank_ms": 0,
                "generate_ms": 0, "total_ms": 0},
        cost={"llm_calls": 0, "embedding_calls": 0, "tokens_in": 0,
              "tokens_out": 0, "est_cost_usd": 0.0},
        answer=None,
        defaults=_defaults(),  # type: ignore[arg-type]
    )
    assert trace["params"]["vector_similarity_weight"] == 0.6
    assert trace["params"]["chunk_token_num"] == 128


def test_build_trace_params_reflect_overrides():
    """params field merges defaults + active overrides. Overrides win."""
    eval_context.set_overrides([
        "vector_similarity_weight=0.4",
        "chunk_token_num=256",
    ])
    trace = eval_context.build_trace(
        question="q", kb="k", mode="vector", top_k=5, level=None,
        retrieved=[],
        timing={"embed_ms": 0, "retrieve_es_ms": 0, "rerank_ms": 0,
                "generate_ms": 0, "total_ms": 0},
        cost={"llm_calls": 0, "embedding_calls": 0, "tokens_in": 0,
              "tokens_out": 0, "est_cost_usd": 0.0},
        answer=None,
        defaults=_defaults(),  # type: ignore[arg-type]
    )
    # Overridden:
    assert trace["params"]["vector_similarity_weight"] == 0.4
    assert trace["params"]["chunk_token_num"] == 256
    # Untouched:
    assert trace["params"]["similarity_threshold"] == 0.1
    assert trace["params"]["local_top_k_seeds"] == 10


def test_build_trace_timestamp_iso_is_well_formed():
    """timestamp_iso must be an ISO-8601 string parseable by datetime."""
    from datetime import datetime
    trace = eval_context.build_trace(
        question="q", kb="k", mode="vector", top_k=5, level=None,
        retrieved=[],
        timing={"embed_ms": 0, "retrieve_es_ms": 0, "rerank_ms": 0,
                "generate_ms": 0, "total_ms": 0},
        cost={"llm_calls": 0, "embedding_calls": 0, "tokens_in": 0,
              "tokens_out": 0, "est_cost_usd": 0.0},
        answer=None,
        defaults=_defaults(),  # type: ignore[arg-type]
    )
    parsed = datetime.fromisoformat(trace["timestamp_iso"])
    assert parsed.tzinfo is not None  # must be timezone-aware


def test_build_trace_does_not_mutate_defaults():
    """The defaults dict the caller passes in must not be mutated by build_trace.

    Otherwise a CLI command that holds a defaults dict across two invocations
    would silently leak overrides from the first into the second."""
    eval_context.set_overrides(["vector_similarity_weight=0.3"])
    defaults = _defaults()
    original = dict(defaults)
    eval_context.build_trace(
        question="q", kb="k", mode="vector", top_k=5, level=None,
        retrieved=[],
        timing={"embed_ms": 0, "retrieve_es_ms": 0, "rerank_ms": 0,
                "generate_ms": 0, "total_ms": 0},
        cost={"llm_calls": 0, "embedding_calls": 0, "tokens_in": 0,
              "tokens_out": 0, "est_cost_usd": 0.0},
        answer=None,
        defaults=defaults,  # type: ignore[arg-type]
    )
    assert defaults == original
