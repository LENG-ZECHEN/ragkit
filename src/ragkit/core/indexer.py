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
    build_graph: bool = False,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict:
    """Index a single file into the named knowledge base.

    Args:
        path: file to index.
        kb_name: target knowledge base.
        build_graph: if True, also extract entities/relations and add to the
            graph for `kb_name`. Slow (one LLM call per chunk) — opt in.
        progress_cb: callable(stage, progress 0..1) for UI updates.

    Returns:
        Summary dict with file/chunks/kb plus graph stats if built.
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
        progress_cb("embedding", 0.4)

    texts = [c["content_with_weight"] for c in chunks]
    vectors = embed_batch(texts)
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(chunks)} chunks"
        )

    # Per-chunk failures show up as None in the vectors list. Tolerate a few,
    # but refuse to silently index a half-vectorized file (users wouldn't know
    # which chunks went missing).
    none_count = sum(1 for v in vectors if v is None)
    if none_count:
        ratio = none_count / len(vectors)
        if ratio > 0.1:
            raise RuntimeError(
                f"Embedding failed for {none_count}/{len(vectors)} chunks ({ratio:.0%}). "
                "Aborting to avoid a partial index. Check API key / quota."
            )
        logger.warning(f"Embedding skipped {none_count}/{len(vectors)} chunks (below 10% threshold)")

    docs = [
        _build_doc(c, kb_name, path.name, v)
        for c, v in zip(chunks, vectors)
        if v is not None
    ]

    if progress_cb:
        progress_cb("indexing", 0.7)

    es = ESConnection()
    es.ensure_index(kb_name)
    errors = es.insert(docs, kb_name)
    if errors:
        logger.error(f"ES insertion errors: {errors}")
        raise RuntimeError(f"Failed to index {path.name}: {errors[:3]}")

    result = {"file": path.name, "chunks": len(docs), "kb": kb_name}

    if build_graph:
        if progress_cb:
            progress_cb("graph_extract", 0.85)
        # Import here so the vector-only path doesn't pay the import cost.
        from ragkit.core.graph.builder import build_graph as build_kb_graph

        # Pass enriched chunks (with id) to the graph builder so source-chunk
        # tracking links back to ES.
        enriched = [
            {"id": d["id"], "content_with_weight": d["content_with_weight"]}
            for d in docs
        ]
        store = build_kb_graph(enriched, kb_name=kb_name, summarize=True)
        result["graph_entities"] = store.entity_count()
        result["graph_relations"] = store.relation_count()

    if progress_cb:
        progress_cb("done", 1.0)

    return result
