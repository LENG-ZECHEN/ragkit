"""Pipeline observability — every visualization in one place.

Two output tiers:

- show_xxx()   — DEFAULT-mode functions, always emit. Surface info that
                 a normal user benefits from seeing on each run.
- trace_xxx()  — DEBUG-mode functions, only emit after enable_debug().
                 Surface internal pipeline state for tuning/debugging.

Business code calls these directly:

    observe.show_chunks_produced("paper.pdf", 87)
    observe.trace_seed_entities(seed_docs)

The functions internally check the debug flag, so call sites stay
clean — no `if debug: print(...)` sprinkled through the pipeline.

Toggling:

    >>> from ragkit.cli import observe
    >>> observe.enable_debug()      # turn debug ON
    >>> observe.disable_debug()     # turn it OFF (mostly used by tests)
    >>> observe.is_debug()
    True

CLI commands flip this from their --debug flag.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterable

from rich.console import Console
from rich.markup import escape as _escape
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


def _pluralize(noun: str, n: int) -> str:
    """English plural that handles the words observe actually uses.

    Specific to local-mode kinds: chunk → chunks, entity → entities,
    community → communities, relation → relations, point → points.
    """
    if n == 1:
        return noun
    irregular = {"entity": "entities", "community": "communities"}
    return irregular.get(noun, noun + "s")

# A single shared Console so output stays interleaved correctly.
console = Console()


# ===========================================================================
# Global debug state
# ===========================================================================


class _ObserverState:
    """Mutable holder for the debug flag (module-level singleton)."""

    __slots__ = ("debug",)

    def __init__(self) -> None:
        self.debug: bool = False


_state = _ObserverState()


def enable_debug() -> None:
    """Turn debug-mode tracing on. Idempotent."""
    _state.debug = True


def disable_debug() -> None:
    """Turn debug-mode tracing off. Mostly for tests."""
    _state.debug = False


def is_debug() -> bool:
    return _state.debug


# ===========================================================================
# Timing helper
# ===========================================================================


@contextmanager
def timed(label: str):
    """Print elapsed wall-clock time after the wrapped block, in debug mode.

    Usage:

        with observe.timed("query embedding"):
            qv = embed_one(question)

    No-op when debug is off (zero overhead beyond the contextmanager
    machinery itself).
    """
    if not is_debug():
        yield
        return

    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        console.print(f"  [dim]⏱  {label}: {elapsed_ms:.0f}ms[/dim]")


# ===========================================================================
# Section 1 — DEFAULT-mode visualizations
# (always emit; useful during normal operation)
# ===========================================================================


def show_chunks_produced(file_name: str, n_chunks: int) -> None:
    """After parse+chunk, surface how many pieces the file became.

    Previously hidden — only visible at the very end as a final count.
    Showing it after parse lets users catch "PDF parsed but produced 0
    chunks" much earlier.
    """
    if n_chunks == 0:
        console.print(f"  [yellow]⚠ {file_name}: produced 0 chunks[/yellow]")
    else:
        console.print(f"  [dim]→ {file_name}: {n_chunks} chunks parsed[/dim]")


def show_dendrogram_structure(level_counts: dict[int, int]) -> None:
    """After clustering, show the community hierarchy at a glance.

    Args:
        level_counts: {level: number_of_communities_at_that_level}
    """
    if not level_counts:
        return
    tree = Tree("[bold]Community hierarchy[/bold]")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        adjective = "coarsest" if level == 0 else "finer" if level == 1 else "finest"
        tree.add(f"Level {level} ({adjective}): [cyan]{count}[/cyan] communities")
    console.print(tree)


def show_es_graph_indexing(stats: dict[str, int]) -> None:
    """After task #24 ES indexing, summarize what got written.

    Args:
        stats: dict with keys entity_embedded, entity_skipped, entity_failed,
               community_embedded, community_failed (from index_graph_to_es).
    """
    parts = []
    if stats.get("entity_embedded"):
        parts.append(f"entities: [green]{stats['entity_embedded']}[/green] new")
    if stats.get("entity_skipped"):
        parts.append(f"[dim]{stats['entity_skipped']} unchanged[/dim]")
    if stats.get("community_embedded"):
        parts.append(f"communities: [green]{stats['community_embedded']}[/green]")
    if stats.get("entity_failed") or stats.get("community_failed"):
        f = stats.get("entity_failed", 0) + stats.get("community_failed", 0)
        parts.append(f"[red]{f} failed[/red]")
    if parts:
        console.print(f"  [dim]ES graph index:[/dim] {' · '.join(parts)}")


def show_retrieval_kind_breakdown(hits: list) -> None:
    """For local/global modes, show how many hits came from each stream.

    Args:
        hits: list of GraphHit objects (have a `.kind` attribute)
    """
    if not hits:
        return
    counts: dict[str, int] = {}
    for h in hits:
        kind = getattr(h, "kind", "?")
        counts[kind] = counts.get(kind, 0) + 1
    parts = [f"{n} {_pluralize(k, n)}" for k, n in counts.items()]
    console.print(f"  [dim]→ {' + '.join(parts)}[/dim]")


def references_table_with_kind(hits: list) -> Table:
    """Build the References table for local mode, with a 'kind' column so
    users see which stream each hit came from."""
    table = Table(title="References", show_lines=False, border_style="dim")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Kind", style="magenta")
    table.add_column("Source")
    table.add_column("Score", justify="right")
    for h in hits:
        kind = getattr(h, "kind", "?")
        title = getattr(h, "title", "?")
        extra = getattr(h, "extra", {}) or {}
        # Score field varies by stream
        score_parts: list[str] = []
        if "similarity" in extra:
            score_parts.append(f"sim={extra['similarity']:.2f}")
        elif "rank" in extra:
            score_parts.append(f"rank={extra['rank']:.1f}")
        elif "weight" in extra:
            score_parts.append(f"w={extra['weight']:.1f}")
        elif "rating" in extra:
            score_parts.append(f"r={extra['rating']}/100")
        elif "source_hits" in extra:
            score_parts.append(f"hits={extra['source_hits']}")
        # Escape title — entity titles like "qwen [model]" would otherwise be
        # interpreted as rich markup and silently lose the [model] part.
        table.add_row(
            str(getattr(h, "rank", "?")),
            kind,
            _escape(title),
            ", ".join(score_parts) or "—",
        )
    return table


# ===========================================================================
# Section 2 — DEBUG-mode tracing (vector retrieval)
# ===========================================================================


def trace_query_rewriting(question: str, queryer: Any) -> None:
    """Re-run the rewriting steps (Step 1-6 of vector mode) and print
    each intermediate stage.

    ``queryer`` is the FulltextQueryer instance from Dealer.qryr; we use
    its tw/syn members + module-level helpers.
    """
    if not is_debug():
        return

    from ragkit.core.rag.nlp import rag_tokenizer
    from ragkit.core.rag.nlp.query import FulltextQueryer

    table = Table(title="Query rewriting (vector mode)", show_lines=True, border_style="cyan")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Output")

    # Step 1 — normalize
    normalized = rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(question.lower()))
    table.add_row("1. Normalize", normalized)

    # Step 2 — strip question/stop words
    cleaned = FulltextQueryer.rmWWW(normalized).strip()
    table.add_row("2. Strip Q-words", cleaned)

    # Step 3 — tokenize (coarse + fine)
    coarse_str = rag_tokenizer.tokenize(cleaned)
    coarse = coarse_str.split()
    fine = rag_tokenizer.fine_grained_tokenize(coarse_str).split()
    table.add_row("3a. Coarse tokens", " · ".join(coarse) or "(empty)")
    table.add_row("3b. Fine tokens", " · ".join(fine) or "(empty)")

    # Step 4 — term weights
    try:
        tks_w = queryer.tw.weights(coarse, preprocess=False)
    except Exception as e:
        tks_w = []
        table.add_row("4. Term weights", f"[red]failed: {e}[/red]")
    if tks_w:
        weights_str = " · ".join(f"{t}^{w:.2f}" for t, w in tks_w[:8])
        table.add_row("4. Term weights", weights_str)

    # Step 5 — synonyms (cap to 5 expansions for readability)
    syn_lines: list[str] = []
    try:
        for tk, _w in tks_w[:5]:
            syn = queryer.syn.lookup(tk)
            if syn:
                syn_lines.append(f"{tk} → {syn}")
    except Exception as e:
        syn_lines = [f"[red]failed: {e}[/red]"]
    if syn_lines:
        table.add_row("5. Synonyms", "\n".join(syn_lines))

    console.print(table)


def trace_vector_retrieval_summary(n_candidates: int, n_returned: int) -> None:
    """After Dealer.retrieval, say how many candidates ES returned and
    how many survived rerank + threshold."""
    if not is_debug():
        return
    console.print(
        f"  [dim]ES → {n_candidates} candidates → rerank → "
        f"[green]{n_returned}[/green] kept[/dim]"
    )


def trace_final_topk_scores(chunks: list) -> None:
    """Print the final score breakdown (similarity / vector / term) per
    surviving chunk."""
    if not is_debug() or not chunks:
        return
    table = Table(title="Final top-K scores", border_style="dim")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Doc")
    table.add_column("Sim", justify="right")
    table.add_column("Vec", justify="right")
    table.add_column("Term", justify="right")
    for c in chunks:
        table.add_row(
            str(getattr(c, "rank", "?")),
            getattr(c, "document_name", "?")[:30],
            f"{getattr(c, 'similarity', 0.0):.3f}",
            f"{getattr(c, 'vector_similarity', 0.0):.3f}",
            f"{getattr(c, 'term_similarity', 0.0):.3f}",
        )
    console.print(table)


# ===========================================================================
# Section 3 — DEBUG-mode tracing (local-mode 4-stream retrieval)
# ===========================================================================


def trace_seed_entities(seed_docs: list[dict]) -> None:
    """Show the entities the vector search found as starting points."""
    if not is_debug() or not seed_docs:
        return
    table = Table(title="Local · seed entities", border_style="cyan")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Source chunks")
    for i, e in enumerate(seed_docs, start=1):
        chunks = e.get("source_chunks_kwd", []) or []
        table.add_row(
            str(i),
            str(e.get("entity_name_kwd", "?")),
            str(e.get("entity_type_kwd", "?")),
            f"{len(chunks)}",
        )
    console.print(table)


def trace_stream_summary(stream_name: str, hits: list) -> None:
    """Generic per-stream tracer used by all 4 local streams."""
    if not is_debug():
        return
    if not hits:
        console.print(f"  [dim]Local · {stream_name}: (empty)[/dim]")
        return
    console.print(f"  [dim]Local · {stream_name}: {len(hits)} hit(s)[/dim]")
    for h in hits[:5]:  # cap for readability
        title = getattr(h, "title", "?")
        # Truncate long titles
        display = title[:60] + "…" if len(title) > 60 else title
        console.print(f"    · {display}")


# ===========================================================================
# Section 4 — DEBUG-mode tracing (global-mode map-reduce)
# ===========================================================================


def trace_global_candidates(community_docs: list[dict]) -> None:
    """Show kNN-retrieved community candidates entering map-reduce."""
    if not is_debug() or not community_docs:
        return
    table = Table(title="Global · candidate community reports", border_style="cyan")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Level", justify="right")
    table.add_column("ID", justify="right")
    table.add_column("Rank", justify="right")
    table.add_column("Preview")
    for i, c in enumerate(community_docs, start=1):
        preview = (c.get("content_with_weight", "") or "").split("\n", 1)[0][:60]
        table.add_row(
            str(i),
            str(c.get("community_level_int", "?")),
            str(c.get("community_id_int", "?")),
            f"{c.get('community_rank_flt', 0.0):.1f}",
            preview,
        )
    console.print(table)


def trace_global_batches(batch_sizes: list[int], total_tokens_per_batch: list[int]) -> None:
    """After token-budget batching, show how the reports got split."""
    if not is_debug():
        return
    parts = [
        f"batch {i}: {sz} reports / ~{tok} tokens"
        for i, (sz, tok) in enumerate(zip(batch_sizes, total_tokens_per_batch), start=1)
    ]
    console.print(f"  [dim]Global · {len(batch_sizes)} batch(es): {' | '.join(parts)}[/dim]")


def trace_global_map_batch(batch_index: int, n_reports: int, rated_points: list) -> None:
    """Per-batch map result: show the rated points produced."""
    if not is_debug():
        return
    if not rated_points:
        console.print(
            f"  [dim]Global · map batch {batch_index} ({n_reports} reports) "
            f"→ [yellow]0 points[/yellow][/dim]"
        )
        return
    console.print(
        f"  [dim]Global · map batch {batch_index} ({n_reports} reports) → "
        f"[green]{len(rated_points)} points[/green]:[/dim]"
    )
    # --debug is opt-in; show all points so the rating distribution is
    # fully inspectable (no silent truncation past the first 3).
    for p in rated_points:
        rating = getattr(p, "rating", 0)
        text = getattr(p, "point", "")
        display = text[:80] + "…" if len(text) > 80 else text
        console.print(f"    · [{rating}/100] {display}")


def trace_global_reduce(n_total_points: int, n_kept: int, threshold: int) -> None:
    """After reduce phase, show how many points survived filtering."""
    if not is_debug():
        return
    console.print(
        f"  [dim]Global · reduce: {n_total_points} points → "
        f"[green]{n_kept}[/green] kept (threshold ≥ {threshold})[/dim]"
    )


# ===========================================================================
# Section 5 — DEBUG-mode tracing (build_graph phases)
# ===========================================================================


def trace_chunk_extraction(chunk_id: str, n_entities: int, n_relations: int) -> None:
    """Per-chunk extraction yield."""
    if not is_debug():
        return
    console.print(
        f"  [dim]extract {chunk_id[:8]}…: "
        f"{n_entities} entities, {n_relations} relations[/dim]"
    )


def trace_consolidation_summary(stats: Any) -> None:
    """Summary after consolidate_all completes.

    Args:
        stats: ConsolidationResult-like object with
               entities_processed / relations_processed / total_calls / failures
    """
    if not is_debug() or stats is None:
        return
    n_entities = len(getattr(stats, "entities_processed", []) or [])
    n_relations = len(getattr(stats, "relations_processed", []) or [])
    calls = getattr(stats, "total_calls", 0)
    failures = getattr(stats, "failures", 0)
    console.print(
        f"  [dim]Consolidation: {n_entities} entities + {n_relations} relations rewritten · "
        f"{calls} LLM calls · {failures} failures[/dim]"
    )


def trace_community_summary_result(community_id: int, title: str, rank: float, n_findings: int) -> None:
    """Per-community report generation result."""
    if not is_debug():
        return
    console.print(
        f"  [dim]Community {community_id}: \"{title[:30]}\" · "
        f"rank {rank:.1f} · {n_findings} findings[/dim]"
    )


# ===========================================================================
# Section 6 — small helpers for CLI commands
# ===========================================================================


def show_panel(title: str, body: str, style: str = "cyan") -> None:
    """Generic helper kept here for convenience (some commands need it)."""
    console.print(Panel(body, title=title, border_style=style))
