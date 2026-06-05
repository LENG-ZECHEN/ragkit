"""Top-level CLI commands. Wired into ragkit.cli.app."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from ragkit.cli.ui import console, error, info, kv_table, success, warn
from ragkit.config import get_config


def cmd_index(
    path: Path = typer.Argument(..., exists=True, help="File or directory to index."),
    kb: str = typer.Option("default", "--kb", "-k", help="Knowledge base name."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into directories."),
    build_graph: bool = typer.Option(
        False,
        "--build-graph",
        help="Also extract entities/relations and build a knowledge graph (slow — one LLM call per chunk).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable pipeline tracing (per-chunk extraction, dendrogram details, ES indexing stats, ...).",
    ),
) -> None:
    """Parse, chunk, embed and index a file or directory into a knowledge base."""
    from ragkit.cli import observe
    from ragkit.core.chunker import is_supported
    from ragkit.core.indexer import index_file

    if debug:
        observe.enable_debug()

    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        files = sorted(p for p in path.glob(pattern) if p.is_file() and is_supported(p))
    else:
        files = [path]

    if not files:
        warn(f"No supported files found at {path}")
        raise typer.Exit(code=1)

    info(
        f"Indexing {len(files)} file(s) into kb=[cyan]{kb}[/cyan]"
        + (" [+graph]" if build_graph else "")
    )

    failures: list[tuple[str, str]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.fields[file]}"),
        BarColumn(),
        TextColumn("{task.fields[stage]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for fp in files:
            task = progress.add_task("", file=fp.name, stage="parsing", total=1.0)

            def cb(stage: str, prog: float) -> None:
                progress.update(task, completed=prog, stage=stage)

            try:
                result = index_file(fp, kb_name=kb, build_graph=build_graph, progress_cb=cb)
                done_label = f"{result['chunks']} chunks"
                if build_graph and "graph_entities" in result:
                    done_label += f", {result['graph_entities']}e/{result['graph_relations']}r"
                progress.update(task, completed=1.0, stage=done_label)
            except Exception as e:
                progress.update(task, completed=1.0, stage="[red]failed[/red]")
                failures.append((fp.name, str(e)))

    if failures:
        error(f"{len(failures)} file(s) failed:")
        for name, msg in failures:
            console.print(f"  • [yellow]{name}[/yellow]: {msg}")
        raise typer.Exit(code=1)

    success(f"Indexed {len(files)} file(s) into {kb}")


def cmd_ask(
    question: str = typer.Argument(..., help="Question to ask."),
    kb: str = typer.Option("default", "--kb", "-k", help="Knowledge base name."),
    top_k: int = typer.Option(5, "--top-k", help="Top chunks to retrieve."),
    mode: str = typer.Option(
        "vector",
        "--mode",
        "-m",
        help="Retrieval mode: vector | local | global.",
    ),
    level: int = typer.Option(
        None,
        "--level",
        help="(global only) Restrict to community hierarchy level N (0=coarsest). "
             "Default: cross-level vector search.",
    ),
    show_thinking: bool = typer.Option(False, "--thinking", help="Stream the LLM's reasoning trace."),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON to stdout."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show internal pipeline trace (query rewriting, kNN candidates, "
             "map/reduce intermediates, ...). For tuning + debugging.",
    ),
) -> None:
    """Ask a single question. Streams the answer to stdout, then prints citations.

    Retrieval modes (Microsoft-GraphRAG-aligned):
      vector  — Original BM25 + dense (default, fastest)
      local   — Entity-centric multi-source retrieval (4 streams: text units,
                community reports, neighbor entities, relations)
      global  — Map-Reduce over community reports (best for thematic queries)
    """
    from ragkit.cli import observe
    from ragkit.core.generator import generate
    from ragkit.core.retriever import retrieve

    if debug:
        observe.enable_debug()

    valid_modes = {"vector", "local", "global"}
    if mode not in valid_modes:
        error(f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(valid_modes))}")
        raise typer.Exit(code=2)

    try:
        if mode == "vector":
            chunks = retrieve(question, kb_name=kb, top_k=top_k)
        else:
            from ragkit.core.graph.retriever import (
                graph_hits_to_chunks,
                retrieve_global,
                retrieve_local,
            )
            if mode == "local":
                hits = retrieve_local(question, kb_name=kb, top_k=top_k)
            else:  # global
                hits = retrieve_global(question, kb_name=kb, top_k=top_k, level=level)
            chunks = graph_hits_to_chunks(hits)
    except Exception as e:
        error(f"Retrieval failed ({mode}): {e}")
        raise typer.Exit(code=2)

    if as_json:
        events = list(generate(question, chunks))
        answer = "".join(e.text for e in events if e.type == "content")
        thinking = "".join(e.text for e in events if e.type == "thinking")
        payload = {
            "question": question,
            "kb": kb,
            "answer": answer,
            "thinking": thinking,
            "references": [c.as_dict() for c in chunks],
        }
        console.print_json(json.dumps(payload, ensure_ascii=False))
        return

    if not chunks:
        warn("No matching chunks in knowledge base — answer may be generic.")
    else:
        console.print(f"\n[dim]Retrieved {len(chunks)} chunk(s) from kb=[cyan]{kb}[/cyan][/dim]\n")

    for event in generate(question, chunks):
        if event.type == "content":
            console.print(event.text, end="", soft_wrap=True, highlight=False)
        elif event.type == "thinking" and show_thinking:
            console.print(event.text, end="", style="dim italic", soft_wrap=True, highlight=False)
        elif event.type == "error":
            console.print()
            error(f"Generation error: {event.text}")
            raise typer.Exit(code=3)

    console.print()
    if chunks:
        console.print()
        if mode == "vector":
            # Vector mode — simple Document/Similarity table.
            table = Table(title="References", show_lines=False, border_style="dim")
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Document")
            table.add_column("Similarity", justify="right")
            for c in chunks:
                table.add_row(str(c.rank), c.document_name, f"{c.similarity:.3f}")
            console.print(table)
        else:
            # local / global — use the kind-aware table from observe so users
            # see which stream each hit came from.
            # `hits` (GraphHit list) is in scope from above.
            console.print(observe.references_table_with_kind(hits))


def cmd_retrieve(
    question: str = typer.Argument(..., help="Question to retrieve for."),
    kb: str = typer.Option("default", "--kb", "-k", help="Knowledge base name."),
    top_k: int = typer.Option(5, "--top-k", help="Top chunks to retrieve."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show query rewriting trace, ES candidates count, rerank timing, ...",
    ),
) -> None:
    """Run retrieval only (no LLM call) — useful for tuning."""
    from ragkit.cli import observe
    from ragkit.core.retriever import retrieve

    if debug:
        observe.enable_debug()

    chunks = retrieve(question, kb_name=kb, top_k=top_k)
    if not chunks:
        warn("No matches.")
        return

    for c in chunks:
        console.rule(
            f"[cyan]#{c.rank}[/cyan] {c.document_name} · "
            f"sim={c.similarity:.3f} (vec={c.vector_similarity:.3f}, term={c.term_similarity:.3f})"
        )
        console.print(c.content)
    console.rule()


def cmd_kb_list() -> None:
    """List all knowledge bases."""
    from ragkit.core.kb_manager import list_kbs

    names = list_kbs()
    if not names:
        info("No knowledge bases yet. Run `rag index` to create one.")
        return
    for n in names:
        console.print(f"  • [cyan]{n}[/cyan]")


def cmd_kb_info(
    name: str = typer.Argument(..., help="Knowledge base name."),
) -> None:
    """Show stats and document list for a knowledge base."""
    from ragkit.core.kb_manager import kb_documents, kb_info

    stats = kb_info(name)
    console.print(kv_table(
        f"KB: {stats.name}",
        [
            ("Documents", str(stats.document_count)),
            ("Chunks", str(stats.chunk_count)),
        ],
    ))

    docs = kb_documents(name)
    if docs:
        t = Table(title="Documents", border_style="dim")
        t.add_column("Name")
        t.add_column("Chunks", justify="right")
        for d in docs:
            t.add_row(d["document_name"], str(d["chunks"]))
        console.print(t)


def cmd_kb_delete(
    name: str = typer.Argument(..., help="Knowledge base name to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a knowledge base. Irreversible.

    Also removes the companion {name}_graph index if it exists (avoids
    orphan graph data after the chunk index is gone).
    """
    from ragkit.core.kb_manager import delete_kb
    from ragkit.core._ragflow.rag.utils.es_conn import ESConnection

    if not yes:
        confirm = typer.confirm(f"Delete knowledge base '{name}'? This cannot be undone.")
        if not confirm:
            info("Cancelled.")
            return

    deleted = delete_kb(name)

    # Also drop the graph companion index — best-effort, may not exist.
    # ISS-022: surface failures at warn level instead of silent except, so a
    # connection / permission issue is visible. Matches graph_cmd.py policy.
    try:
        ESConnection().delete_index(f"{name}_graph")
    except Exception as e:
        warn(f"Could not delete companion graph index '{name}_graph': {e}")

    if deleted:
        success(f"Deleted '{name}'")
    else:
        warn(f"'{name}' did not exist")


def cmd_doctor() -> None:
    """Verify config and connections (ES, API key, dict files)."""
    import os
    from ragkit.core._ragflow.rag.utils.es_conn import ESConnection

    cfg = get_config()
    ok = True

    # API key
    if cfg.dashscope_api_key:
        success(f"DASHSCOPE_API_KEY set ({len(cfg.dashscope_api_key)} chars)")
    else:
        error("DASHSCOPE_API_KEY not set — copy .env.example to .env")
        ok = False

    # ES
    try:
        es = ESConnection()
        if es.es.ping():
            success(f"Elasticsearch reachable at {cfg.es_host}")
        else:
            error(f"Elasticsearch ping failed at {cfg.es_host}")
            ok = False
    except Exception as e:
        error(f"Elasticsearch connection failed: {e}")
        ok = False

    # Tokenizer dict
    from ragkit.core._ragflow.api.utils.file_utils import get_project_base_directory
    dict_path = os.path.join(get_project_base_directory(), "rag", "res", "huqie.txt")
    if os.path.exists(dict_path):
        size_mb = os.path.getsize(dict_path) / 1024 / 1024
        success(f"Tokenizer dict found ({size_mb:.1f} MB)")
    else:
        error(f"Tokenizer dict missing at {dict_path}")
        ok = False

    # Model identifiers
    info(f"LLM model: {cfg.llm_model}")
    info(f"Embedding model: {cfg.embedding_model} (dim={cfg.embedding_dim})")

    raise typer.Exit(code=0 if ok else 1)
