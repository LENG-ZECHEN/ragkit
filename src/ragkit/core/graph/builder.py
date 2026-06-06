"""Orchestrates graph construction: extract → consolidate → detect → summarize.

This is the function the indexer (and `rag graph build` CLI) call.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from ragkit.core.graph.community import detect_communities
from ragkit.core.graph.description_merger import consolidate_all
from ragkit.core.graph.es_indexer import index_graph_to_es
from ragkit.core.graph.extractor import extract_from_text
from ragkit.core.graph.store import GraphStore, NetworkXGraphStore, open_store
from ragkit.core.graph.summarizer import summarize_all
from ragkit.logger import logger


# Bounded worker pool for per-chunk LLM extraction. DashScope rate limit
# is the binding constraint; 5 is a safe default for small-to-medium batches.
MAX_CONCURRENT_EXTRACTIONS = 5


def build_graph(
    chunks: Iterable[dict],
    kb_name: str,
    *,
    summarize: bool = True,
    consolidate_descriptions: bool = True,
    index_to_es: bool = True,
    max_summary_communities: int = 20,
    max_consolidation_calls: int = 20,
    progress_cb: Callable[[str, int, int], None] | None = None,
    store: GraphStore | None = None,
) -> GraphStore:
    """Build (or extend) the knowledge graph for a KB.

    Args:
        chunks: iterable of dicts with at least 'id' and 'content_with_weight'.
        kb_name: knowledge base name (graph file is per-KB).
        summarize: if True, run community summaries (slow — many LLM calls).
        max_summary_communities: cap on how many communities we summarize.
        progress_cb: callable(stage, current, total) for UI updates.
        store: optional override (mostly for tests).

    Returns:
        The populated and persisted GraphStore.
    """
    if store is None:
        store = open_store(kb_name)

    chunks = list(chunks)
    total = len(chunks)
    if total == 0:
        logger.warning("No chunks given to build_graph")
        return store

    from ragkit.cli import observe

    # ---- 1. extract entities + relations per chunk -------------------
    # Two-stage to make this concurrent-safe:
    #   Stage A: concurrent LLM extraction (pure function, no graph mutation)
    #   Stage B: serial merge into the graph (NetworkX is NOT thread-safe;
    #            upsert_entity/relation mutate shared store state)

    def _extract_one(chunk: dict) -> tuple[str, str, object]:
        """Stage-A worker: call LLM, return (chunk_id, text, result).
        Pure — no shared-state writes."""
        text = chunk.get("content_with_weight", "")
        chunk_id = str(chunk.get("id", ""))
        return chunk_id, text, extract_from_text(text, chunk_id)

    extraction_results: list[tuple[str, str, object]] = []
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENT_EXTRACTIONS, total)
    ) as executor:
        # Preserve the original chunk numbering for fallback chunk_id only.
        numbered_chunks = list(enumerate(chunks, start=1))
        futures = {
            executor.submit(_extract_one, chunk): i for i, chunk in numbered_chunks
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if progress_cb:
                progress_cb("extracting", completed, total)
            i = futures[future]
            try:
                chunk_id, text, result = future.result()
            except Exception as e:
                # Per-chunk failure: log + count as failure, continue with others
                logger.warning(f"Extraction failed for chunk #{i}: {e}")
                chunk_id, text, result = (
                    str(numbered_chunks[i - 1][1].get("id", f"chunk-{i}")),
                    numbered_chunks[i - 1][1].get("content_with_weight", ""),
                    None,
                )
            extraction_results.append((chunk_id, text, result))

    # Stage B: serial merge — guaranteed thread-safe (main thread only)
    extraction_failures = 0
    for chunk_id, text, result in extraction_results:
        if result is None:
            # Extraction threw — treat as failure if chunk had content
            if text.strip():
                extraction_failures += 1
            continue
        observe.trace_chunk_extraction(chunk_id, len(result.entities), len(result.relations))
        for entity in result.entities:
            store.upsert_entity(entity)
        for relation in result.relations:
            store.upsert_relation(relation)
        # If a non-empty chunk produced ZERO entities AND ZERO relations,
        # extractor likely hit an API error (already logged) — count it.
        if text.strip() and not result.entities and not result.relations:
            extraction_failures += 1

    # If most extractions failed, the graph is bogus — refuse to persist
    # an empty graph that would mask the real problem (API down, bad key).
    failure_ratio = extraction_failures / total if total else 0
    if failure_ratio > 0.5 and store.entity_count() == 0:
        raise RuntimeError(
            f"Graph extraction failed on {extraction_failures}/{total} chunks. "
            "Check API key / quota / network. Refusing to save an empty graph."
        )

    logger.info(
        f"Extracted {store.entity_count()} entities, {store.relation_count()} relations "
        f"({extraction_failures} chunk(s) produced nothing)"
    )

    # ---- 1.5. description consolidation (LLM rewrite of long descriptions) ----
    if consolidate_descriptions:
        result_consolidation = consolidate_all(
            store,
            max_calls=max_consolidation_calls,
            progress_cb=progress_cb,
        )
        observe.trace_consolidation_summary(result_consolidation)

    # ---- 2. community detection --------------------------------------
    if isinstance(store, NetworkXGraphStore):
        if progress_cb:
            progress_cb("clustering", 1, 1)
        communities = detect_communities(store)
        store.set_communities(communities)
        # Default-mode: surface the hierarchical structure (always visible).
        level_counts: dict[int, int] = {}
        for c in communities:
            level_counts[c.level] = level_counts.get(c.level, 0) + 1
        observe.show_dendrogram_structure(level_counts)

    # ---- 3. community summaries (progress now reflects real work) ----
    if summarize and store.all_communities():
        summarize_all(
            store,
            max_communities=max_summary_communities,
            progress_cb=progress_cb,
        )
        # Debug-mode trace: per-community report stats
        for c in store.all_communities()[:max_summary_communities or len(store.all_communities())]:
            observe.trace_community_summary_result(c.id, c.title, c.rank, len(c.findings))

    store.save()

    # ---- 4. ES indexing of graph artifacts (task #24) ----------------
    if index_to_es:
        try:
            es_stats = index_graph_to_es(store, kb_name, progress_cb=progress_cb)
            observe.show_es_graph_indexing(es_stats)
        except Exception as e:
            # ES indexing failures shouldn't lose the JSON graph (already
            # saved above). Log loudly so the user knows graph retrieval
            # via ES won't work until they fix ES + re-run build.
            logger.error(
                f"Graph ES indexing failed for {kb_name}: {e}. "
                "JSON graph is still saved; rerun `rag graph build` after fixing ES."
            )

    return store
