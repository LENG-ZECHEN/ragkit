"""REPL state machine — slash commands users actually type.

Phase A (task #27) added /mode, /level, /debug and made ReplState carry
mode/debug/level fields. The tests below cover both the original
commands and the new ones.
"""

from __future__ import annotations

from ragkit.cli import observe
from ragkit.cli.repl import HELP_TEXT, VALID_MODES, ReplState, _handle_command
from ragkit.core.retriever import RetrievedChunk


import pytest


@pytest.fixture(autouse=True)
def reset_observe_state():
    """Each test starts with observe debug OFF — /debug toggles can leak
    between tests otherwise."""
    observe.disable_debug()
    yield
    observe.disable_debug()


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


def test_show_valid_index_returns_unchanged_state(capsys):
    """`/show 1` on a valid index must NOT mutate state (only print)."""
    chunks = [
        RetrievedChunk(
            rank=1, document_id="d", document_name="report.pdf",
            content="MARKER_TEXT_FOR_SHOW", similarity=0.9,
            vector_similarity=0.9, term_similarity=0.9,
        ),
    ]
    s_in = _state(last_chunks=chunks, top_k=7, kb="finance")
    s_out = _handle_command("/show 1", s_in)
    # State is unchanged — /show is a query, not a mutation.
    assert s_out.kb == "finance"
    assert s_out.top_k == 7
    assert s_out.last_chunks == chunks
    # The output went somewhere — content is printed via rich (captured via capsys).
    captured = capsys.readouterr()
    assert "MARKER_TEXT_FOR_SHOW" in captured.out or "MARKER_TEXT_FOR_SHOW" in captured.err


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


# ==========================================================================
# /mode command (task #27)
# ==========================================================================


def test_mode_default_is_vector():
    """ReplState's default mode must be vector — the safest choice."""
    s = ReplState(kb="x", top_k=5, show_thinking=False, last_chunks=[])
    assert s.mode == "vector"


def test_mode_accepts_each_valid_value():
    """Each of vector/local/global must be settable."""
    s = _state()
    for valid in VALID_MODES:
        new = _handle_command(f"/mode {valid}", s)
        assert new is not None
        assert new.mode == valid


def test_mode_rejects_invalid_value():
    s = _state(mode="vector")
    new = _handle_command("/mode telepathy", s)
    # State must be unchanged on rejection.
    assert new.mode == "vector"


def test_mode_rejects_missing_arg():
    s = _state(mode="vector")
    new = _handle_command("/mode", s)
    assert new.mode == "vector"


def test_mode_leaving_global_clears_level():
    """A level filter only makes sense for global mode; switching away
    must clear it so a later /mode global doesn't carry stale state."""
    s = _state(mode="global", level=2)
    new = _handle_command("/mode vector", s)
    assert new.mode == "vector"
    assert new.level is None


def test_mode_within_global_keeps_level():
    """Setting mode to global again shouldn't drop the level."""
    s = _state(mode="global", level=2)
    new = _handle_command("/mode global", s)
    assert new.level == 2


# ==========================================================================
# /level command
# ==========================================================================


def test_level_default_is_none():
    s = ReplState(kb="x", top_k=5, show_thinking=False, last_chunks=[])
    assert s.level is None


def test_level_accepts_non_negative_integer():
    s = _state(mode="global")
    new = _handle_command("/level 2", s)
    assert new.level == 2


def test_level_none_clears():
    s = _state(mode="global", level=2)
    new = _handle_command("/level none", s)
    assert new.level is None


def test_level_zero_clears():
    """`/level 0` and `/level none` both treated as 'no filter' — 0 is
    treated as the sentinel here, not as 'coarsest level'. We use None
    for 'unset' to avoid confusion with level=0=coarsest."""
    s = _state(mode="global", level=5)
    new = _handle_command("/level 0", s)
    assert new.level is None


def test_level_rejects_negative():
    s = _state(mode="global", level=2)
    new = _handle_command("/level -3", s)
    # Reject silently (warning printed but state unchanged).
    assert new.level == 2


def test_level_in_non_global_mode_still_sets_but_warns():
    """Setting level outside global mode is allowed (user may switch later)
    but should warn — we don't crash."""
    s = _state(mode="vector", level=None)
    new = _handle_command("/level 1", s)
    assert new.level == 1
    # Mode should not have been changed.
    assert new.mode == "vector"


# ==========================================================================
# /debug command
# ==========================================================================


def test_debug_default_off():
    s = ReplState(kb="x", top_k=5, show_thinking=False, last_chunks=[])
    assert s.debug is False
    assert observe.is_debug() is False


def test_debug_toggle_syncs_observe():
    """/debug must keep ReplState and observe module in sync."""
    s = _state(debug=False)
    new = _handle_command("/debug", s)
    assert new.debug is True
    assert observe.is_debug() is True

    # Toggle off
    new2 = _handle_command("/debug", new)
    assert new2.debug is False
    assert observe.is_debug() is False


def test_debug_toggle_idempotent_via_repeated_invocations():
    """Two toggles return us to the original state."""
    s = _state(debug=False)
    after_two = _handle_command("/debug", _handle_command("/debug", s))
    assert after_two.debug is False
    assert observe.is_debug() is False


# ==========================================================================
# HELP_TEXT covers the new commands
# ==========================================================================


def test_help_text_lists_all_commands():
    """If we add a slash command, the help text must mention it. This
    test is a cheap forcing function to keep help in sync."""
    expected_in_help = ["/kb", "/mode", "/level", "/top", "/thinking",
                        "/debug", "/show", "/clear", "/help", "/exit"]
    for cmd in expected_in_help:
        assert cmd in HELP_TEXT, f"{cmd} missing from HELP_TEXT"
