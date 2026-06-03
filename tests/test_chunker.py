"""Chunker dispatch and validation.

We don't test the deep parsing logic (that's third-party); we test the
contract our chunker presents: supported extensions, error cases, and
that small text files produce non-trivial output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.chunker import SUPPORTED_EXTS, chunk_file, is_supported


def test_is_supported_recognizes_common_formats():
    assert is_supported(Path("foo.pdf"))
    assert is_supported(Path("foo.PDF"))  # case-insensitive
    assert is_supported(Path("a/b/c.docx"))
    assert is_supported(Path("notes.md"))


def test_is_supported_rejects_unknown_formats():
    assert not is_supported(Path("foo.zip"))
    assert not is_supported(Path("foo.exe"))
    assert not is_supported(Path("foo"))  # no extension


def test_supported_exts_includes_critical_formats():
    """Regression guard — these are the formats users expect to work."""
    must_have = {".pdf", ".docx", ".txt", ".md", ".html", ".xlsx"}
    assert must_have.issubset(SUPPORTED_EXTS)


def test_chunk_file_raises_on_missing_file(tmp_path):
    """Helpful error when path doesn't exist (common user mistake)."""
    with pytest.raises(FileNotFoundError):
        chunk_file(tmp_path / "does-not-exist.pdf")


def test_chunk_file_raises_on_unsupported_type(tmp_path):
    """Reject early with a clear message rather than crashing deep in parsers."""
    bogus = tmp_path / "bad.xyz"
    bogus.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        chunk_file(bogus)


def test_chunk_file_produces_chunks_from_txt(sample_txt):
    """End-to-end: small Chinese text file produces at least one chunk
    with expected fields populated."""
    chunks = chunk_file(sample_txt, chunk_token_num=32)

    assert len(chunks) >= 1
    first = chunks[0]
    # The downstream indexer relies on these exact keys:
    for required_field in ("content_with_weight", "content_ltks", "content_sm_ltks", "docnm_kwd"):
        assert required_field in first, f"Missing {required_field} in chunk"

    # Chunk content should contain text from the input file.
    all_content = "".join(c["content_with_weight"] for c in chunks)
    assert "人工智能" in all_content
    assert "RAG" in all_content
