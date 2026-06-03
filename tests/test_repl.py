"""REPL state machine — slash commands users actually type."""

from __future__ import annotations

from ragkit.cli.repl import ReplState, _handle_command
from ragkit.core.retriever import RetrievedChunk


def _state(**overrides) -> ReplState:
    base = dict(kb="default", top_k=5, show_thinking=False, last_chunks=[])
    base.update(overrides)
    return ReplState(**base)


def test_exit_command_returns_none():
    """The REPL loop watches for None to break out."""
    assert _handle_command("/exit", _state()) is None
    assert _handle_command("/quit", _state()) is None
    assert _handle_command("/q", _state()) is None


def test_kb_command_switches_knowledge_base():
    new_state = _handle_command("/kb finance", _state(kb="default"))
    assert new_state is not None
    assert new_state.kb == "finance"


def test_kb_command_missing_arg_is_noop():
    """Don't crash on `/kb` with no name; just print an error."""
    new = _handle_command("/kb", _state(kb="default"))
    assert new.kb == "default"  # unchanged


def test_top_command_updates_top_k():
    new = _handle_command("/top 10", _state(top_k=5))
    assert new.top_k == 10


def test_top_command_rejects_out_of_range():
    """top_k of 0 or 1000 would burn the user's quota — clamp at sane bounds."""
    new = _handle_command("/top 0", _state(top_k=5))
    assert new.top_k == 5  # unchanged

    new = _handle_command("/top 999", _state(top_k=5))
    assert new.top_k == 5  # unchanged

    new = _handle_command("/top notanumber", _state(top_k=5))
    assert new.top_k == 5


def test_thinking_command_toggles():
    """Toggle behavior — not set-to-on, so users can type it twice safely."""
    s1 = _handle_command("/thinking", _state(show_thinking=False))
    assert s1.show_thinking is True

    s2 = _handle_command("/thinking", s1)
    assert s2.show_thinking is False


def test_show_without_chunks_is_safe():
    """User types /show 1 before asking anything — must not raise."""
    s = _handle_command("/show 1", _state(last_chunks=[]))
    assert s is not None  # didn't exit, didn't crash


def test_show_out_of_range_is_safe():
    chunks = [
        RetrievedChunk(rank=1, document_id="x", document_name="x", content="x",
                       similarity=0.0, vector_similarity=0.0, term_similarity=0.0),
    ]
    s = _handle_command("/show 99", _state(last_chunks=chunks))
    assert s is not None  # didn't exit


def test_unknown_command_returns_unchanged_state():
    """Unknown /commands print help — they don't exit or mutate state."""
    original = _state(kb="finance", top_k=7)
    new = _handle_command("/unknown_command_xyz", original)
    assert new is not None
    assert new.kb == "finance"
    assert new.top_k == 7


def test_state_is_immutable_on_no_op_commands():
    """The /clear and /help commands must not silently mutate state."""
    s = _state(kb="finance", top_k=7)
    # /help shouldn't change kb / top_k
    after_help = _handle_command("/help", s)
    assert after_help.kb == "finance"
    assert after_help.top_k == 7
