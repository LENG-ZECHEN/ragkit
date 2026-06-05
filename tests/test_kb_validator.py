"""Tests for kb name validation (ISS-003 — path traversal defense)."""

from __future__ import annotations

import pytest

from ragkit.core.kb_validator import validate_kb_name


# ===== Valid names =========================================================


@pytest.mark.parametrize(
    "name",
    [
        "default",
        "finance",
        "kb1",
        "my-kb",
        "my_kb",
        "a",
        "1",
        "a" * 63,  # boundary: exactly max length
        "test123_abc-xyz",
    ],
)
def test_validate_accepts_valid_names(name):
    """Lowercase + digits + _ + - and within 63 chars all pass."""
    validate_kb_name(name)  # no raise


# ===== Rejected: path-traversal patterns ===================================


@pytest.mark.parametrize(
    "name",
    [
        "../etc/passwd",
        "..",
        "../../foo",
        "foo/../bar",
        "/etc/passwd",
        "foo/bar",
        "\\windows\\system32",
        "a/b",
    ],
)
def test_validate_rejects_path_traversal(name):
    """Any name with /, \\, .. must be rejected."""
    with pytest.raises(ValueError, match="Invalid kb name"):
        validate_kb_name(name)


# ===== Rejected: uppercase / spaces / specials ==============================


@pytest.mark.parametrize(
    "name",
    [
        "MyKB",       # uppercase
        "MYKB",       # all caps
        "Test",       # leading upper
        "my kb",      # space
        " kb",        # leading space
        "kb ",        # trailing space
        "kb!",        # punctuation
        "kb@team",    # @
        "kb#1",       # #
        "kb.json",    # dot
        "kb,kb",      # comma
        "kb*",        # star
    ],
)
def test_validate_rejects_special_chars(name):
    """ES-incompatible characters must be rejected."""
    with pytest.raises(ValueError, match="Invalid kb name"):
        validate_kb_name(name)


# ===== Rejected: structural problems ========================================


@pytest.mark.parametrize(
    "name",
    [
        "",           # empty
        "_leading",   # ES forbids _ start
        "-leading",   # ES forbids - start
        "a" * 64,     # one over max length
        "a" * 100,    # way over
    ],
)
def test_validate_rejects_structural_problems(name):
    """Empty / wrong-start / too-long names must be rejected."""
    with pytest.raises(ValueError, match="Invalid kb name"):
        validate_kb_name(name)


# ===== Rejected: non-string input ===========================================


@pytest.mark.parametrize("name", [None, 123, ["kb"], {"kb": "x"}, b"kb"])
def test_validate_rejects_non_string(name):
    """Defensive against bad-type input (e.g., bytes from accidental encoding)."""
    with pytest.raises(ValueError, match="Invalid kb name"):
        validate_kb_name(name)  # type: ignore[arg-type]
