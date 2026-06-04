"""CLI commands for the graph layer: build / info / show / clear."""

from __future__ import annotations

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ragkit.cli.ui import console, error, info, kv_table, success, warn
from ragkit.logger import logger


def cmd_graph_build(
    kb: str = typer.Option(..., "--kb", "-k", help="Knowledge base name (existing ES index)."),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Run community summaries (slow)."),
    max_summaries: int = typer.Option(20, "--max-summaries", help="Cap on summarized communities."),
    consolidate: bool = typer.Option(
        True,
        "--consolidate/--no-consolidate",
        help="LLM-merge long entity/relation descriptions to keep them concise.",
    ),
    max_consolidations: int = typer.Option(
        20,
        "--max-consolidations",
        help="Cap on consolidation LLM calls per build.",
    ),
) -> None:
    """Build a knowledge graph from an already-indexed KB.

    Reads chunks back from Elasticsearch and runs entity/relation extraction.
    """
    from ragkit.core.graph.builder import build_graph
    from ragkit.core.rag.utils.es_conn import ESConnection

    if not kb or not kb.strip():
        error("--kb must be a non-empty knowledge base name")
        raise typer.Exit(code=2)

    es = ESConnection().es
    if not es.indices.exists(index=kb):
        error(f"Knowledge base '{kb}' does not exist. Index documents first with `rag index`.")
        raise typer.Exit(code=1)

    # Scroll all chunks from ES.
    info(f"Scanning chunks from kb=[cyan]{kb}[/cyan]...")
    chunks: list[dict] = []
    resp = es.search(
        index=kb,
        body={"query": {"match_all": {}}, "_source": ["content_with_weight"]},
        size=1000,
        scroll="2m",
    )
    sid = resp.get("_scroll_id")
    try:
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                content = src.get("content_with_weight", "")
                if content.strip():
                    chunks.append({"id": h["_id"], "content_with_weight": content})
            resp = es.scroll(scroll_id=sid, scroll="2m")
            sid = resp.get("_scroll_id", sid)
    finally:
        if sid:
            try:
                es.clear_scroll(scroll_id=sid)
            except Exception as e:
                # Leave a breadcrumb — leaking a scroll cursor isn't fatal
                # (ES TTL cleans up) but a real connection problem here is
                # useful diagnostic info.
                logger.debug(f"clear_scroll failed: {e}")

    if not chunks:
        warn(f"No chunks found in {kb}")
        raise typer.Exit(code=1)

    info(f"Building graph from {len(chunks)} chunks (one LLM call per chunk for extraction)")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.fields[stage]}"),
        BarColumn(),
        TextColumn("{task.fields[detail]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("build", stage="starting", detail="", total=1.0)

        def cb(stage: str, current: int, total: int) -> None:
            progress.update(
                task,
                completed=current / max(total, 1),
                stage=stage,
                detail=f"{current}/{total}",
            )

        store = build_graph(
            chunks,
            kb_name=kb,
            summarize=summarize,
            consolidate_descriptions=consolidate,
            max_summary_communities=max_summaries,
            max_consolidation_calls=max_consolidations,
            progress_cb=cb,
        )
        progress.update(task, completed=1.0, stage="done", detail="")

    success(
        f"Graph built for kb=[cyan]{kb}[/cyan]: "
        f"{store.entity_count()} entities, {store.relation_count()} relations, "
        f"{len(store.all_communities())} communities"
    )


def cmd_graph_info(
    kb: str = typer.Argument(..., help="Knowledge base name."),
) -> None:
    """Show graph stats for a knowledge base."""
    from ragkit.core.graph.store import open_store

    store = open_store(kb)
    if store.entity_count() == 0:
        warn(f"No graph for '{kb}'. Run `rag graph build --kb {kb}` first.")
        return

    type_counts: dict[str, int] = {}
    for e in store.all_entities():
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    console.print(kv_table(
        f"Graph: {kb}",
        [
            ("Entities", str(store.entity_count())),
            ("Relations", str(store.relation_count())),
            ("Communities", str(len(store.all_communities()))),
        ],
    ))

    t = Table(title="Entities by type", border_style="dim")
    t.add_column("Type", style="cyan")
    t.add_column("Count", justify="right")
    for tp, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        t.add_row(tp, str(cnt))
    console.print(t)


def cmd_graph_show(
    kb: str = typer.Argument(..., help="Knowledge base name."),
    entity: str = typer.Argument(..., help="Entity name to inspect."),
    depth: int = typer.Option(1, "--depth", "-d", help="BFS neighborhood depth."),
) -> None:
    """Show one entity and its neighborhood."""
    from ragkit.core.graph.store import open_store

    store = open_store(kb)
    e = store.get_entity(entity)
    if not e:
        error(f"Entity '{entity}' not found in graph for '{kb}'")
        raise typer.Exit(code=1)

    console.print(kv_table(
        f"Entity: {e.name}",
        [
            ("Type", e.type),
            ("Description", e.description or "(none)"),
            ("Mentions", str(len(e.source_chunks))),
        ],
    ))

    neighbors = store.neighbors(e.name, depth=depth)
    if not neighbors:
        info(f"No neighbors within depth {depth}.")
        return

    t = Table(title=f"Neighbors (depth={depth})", border_style="dim")
    t.add_column("Name", style="cyan")
    t.add_column("Type")
    t.add_column("Description")
    for nb in neighbors[:30]:
        t.add_row(nb.name, nb.type, (nb.description or "")[:80])
    console.print(t)


def cmd_graph_clear(
    kb: str = typer.Argument(..., help="Knowledge base name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete the graph file for a knowledge base."""
    from ragkit.core.graph.store import open_store

    if not yes:
        if not typer.confirm(f"Delete graph for '{kb}'? (vector index stays)"):
            info("Cancelled.")
            return
    store = open_store(kb)
    store.clear()
    success(f"Cleared graph for '{kb}'")
