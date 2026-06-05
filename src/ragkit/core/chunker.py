"""Chunking facade.

Routes a file to the right deepdoc/rag parser by extension, then merges
sections into token-bounded chunks. Returns chunks ready for embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ragkit.core._ragflow.rag.app.naive import chunk as _naive_chunk
from ragkit.logger import logger


SUPPORTED_EXTS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".txt", ".md", ".markdown", ".html", ".htm",
    ".json", ".ppt", ".pptx",
    # Source-code extensions handled by TxtParser inside naive.chunk
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".sh", ".sql",
}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def chunk_file(
    path: Path,
    chunk_token_num: int = 128,
    delimiter: str = "\n!?。；！？",
    layout_recognize: str = "DeepDOC",
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """Parse a file and return a list of chunk dicts.

    Each dict has at least:
        docnm_kwd       — file name
        title_tks       — tokenized file name
        content_with_weight — chunk text
        content_ltks    — coarse-grained tokens
        content_sm_ltks — fine-grained tokens
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not is_supported(path):
        raise ValueError(
            f"Unsupported file type {path.suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTS)}"
        )

    def _cb(prog: float | None = None, msg: str = "") -> None:
        if progress_cb and msg:
            progress_cb(prog or 0.0, msg)
        elif msg:
            logger.debug(f"[parse] {msg}")

    parser_config = {
        "chunk_token_num": chunk_token_num,
        "delimiter": delimiter,
        "layout_recognize": layout_recognize,
    }

    chunks = _naive_chunk(
        filename=str(path),
        binary=None,
        from_page=0,
        to_page=100_000,
        lang="Chinese",
        callback=_cb,
        parser_config=parser_config,
    )
    logger.info(f"Parsed {path.name} → {len(chunks)} chunks")
    return chunks
