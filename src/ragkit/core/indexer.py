"""Ingestion pipeline: parse → chunk → embed → write to ES."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable

import xxhash

from ragkit.config import get_config
from ragkit.core.chunker import chunk_file
from ragkit.core.embedder import embed_batch
from ragkit.core.rag.utils.es_conn import ESConnection
from ragkit.logger import logger


def _build_doc(chunk: dict, kb_name: str, file_name: str, embedding: list[float]) -> dict:
    """Assemble one ES document from a chunk and its embedding vector."""
    content = chunk["content_with_weight"]
    chunk_id = xxhash.xxh64((content + kb_name).encode("utf-8")).hexdigest()
    doc_id = xxhash.xxh64(file_name.encode("utf-8")).hexdigest()
    now = datetime.datetime.now()

    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "docnm": file_name,
        "docnm_kwd": chunk.get("docnm_kwd", file_name),
        "title_tks": chunk.get("title_tks", ""),
        "title_sm_tks": chunk.get("title_sm_tks", ""),
        "content_ltks": chunk["content_ltks"],
        "content_sm_ltks": chunk["content_sm_ltks"],
        "content_with_weight": content,
        "important_kwd": [],
        "important_tks": [],
        "question_kwd": [],
        "question_tks": [],
        "kb_id": kb_name,
        "create_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "create_timestamp_flt": now.timestamp(),
        f"q_{len(embedding)}_vec": embedding,
    }


def index_file(
    path: Path,
    kb_name: str,
    *,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict:
    """Index a single file into the named knowledge base.

    Returns a summary dict: {file, chunks, kb}.
    """
    cfg = get_config()
    cfg.require_api_key()

    path = Path(path).resolve()
    if progress_cb:
        progress_cb("parsing", 0.1)

    chunks = chunk_file(path)
    if not chunks:
        logger.warning(f"No chunks produced from {path}")
        return {"file": path.name, "chunks": 0, "kb": kb_name}

    if progress_cb:
        progress_cb("embedding", 0.5)

    texts = [c["content_with_weight"] for c in chunks]
    vectors = embed_batch(texts)
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks"
        )

    docs = [
        _build_doc(c, kb_name, path.name, v)
        for c, v in zip(chunks, vectors)
        if v is not None
    ]

    if progress_cb:
        progress_cb("indexing", 0.9)

    es = ESConnection()
    es.ensure_index(kb_name)
    errors = es.insert(docs, kb_name)
    if errors:
        logger.error(f"ES insertion errors: {errors}")
        raise RuntimeError(f"Failed to index {path.name}: {errors[:3]}")

    if progress_cb:
        progress_cb("done", 1.0)

    return {"file": path.name, "chunks": len(docs), "kb": kb_name}
