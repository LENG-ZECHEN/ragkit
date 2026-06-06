"""Top-level CLI commands. Wired into ragkit.cli.app."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from ragkit.cli.ui import console, error, info, kv_table, success, warn
from ragkit.config import get_config
from ragkit.core.kb_validator import validate_kb_name


# --------------------------------------------------------------------------
# Eval trace helpers (Phase 0 of the eval instrumentation layer).
# --------------------------------------------------------------------------


def _retrieved_items_from_chunks(chunks: list, hits: list | None) -> list[dict[str, Any]]:
    """Build the ``retrieved`` array for an EvalTrace.

    Prefers GraphHit data when available (so ``kind`` is accurate); falls back
    to RetrievedChunk data (kind always "chunk") for vector mode.
    """
    if hits is not None:
        out: list[dict[str, Any]] = []
        for h in hits:
            extra = getattr(h, "extra", {}) or {}
            score = float(
                extra.get("similarity",
                          extra.get("rating",
                                    extra.get("weight",
                                              extra.get("rank", 0.0)))) or 0.0
            )
            out.append({
                "chunk_id": str(extra.get("chunk_id") or extra.get("name") or h.title),
                "rank": int(h.rank),
                "score": score,
                "kind": h.kind,
            })
        return out
    return [
        {
            "chunk_id": c.chunk_id,
            "rank": int(c.rank),
            "score": float(c.similarity),
            "kind": "chunk",
        }
        for c in chunks
    ]


def _emit_eval_trace(
    *,
    question: str,
    kb: str,
    mode: str,
    top_k: int,
    level: int | None,
    chunks: list,
    hits: list | None,
    timing: dict[str, float],
    answer: str | None,
    llm_calls: int,
    eval_out: Path | None,
) -> None:
    """Assemble + emit the EvalTrace JSON. Internal helper used by cmd_ask
    and cmd_retrieve.

    Output: if ``eval_out`` is given, writes to that file; else prints a
    compact one-line JSON to stdout (so a downstream consumer can pipe it
    straight into ``jq``).
    """
    from ragkit import eval_context
    from ragkit.core.graph import global_search as gs
    from ragkit.core.graph import retriever as graph_ret

    defaults: dict[str, Any] = {
        "vector_similarity_weight": 0.6,
        "similarity_threshold": 0.1,
        "chunk_token_num": 128,
        "local_top_k_seeds": graph_ret.LOCAL_TOP_K_SEEDS,
        "local_top_k_text_units": graph_ret.LOCAL_TOP_K_TEXT_UNITS,
        "local_top_k_communities": graph_ret.LOCAL_TOP_K_COMMUNITIES,
        "local_top_k_entities": graph_ret.LOCAL_TOP_K_ENTITIES,
        "local_top_k_relations": graph_ret.LOCAL_TOP_K_RELATIONS,
        "global_top_k_reports": graph_ret.GLOBAL_TOP_K_REPORTS,
        "map_batch_token_budget": gs.MAP_BATCH_TOKEN_BUDGET,
        "rating_threshold": gs.RATING_THRESHOLD,
        "default_final_top_k": gs.DEFAULT_FINAL_TOP_K,
    }

    # Cost is best-effort. tokens_in: a rough proxy from the prompt the
    # retriever sees (question chars / 2). tokens_out: CJK-friendly heuristic.
    answer_str = answer or ""
    cost: dict[str, Any] = {
        "llm_calls": int(llm_calls),
        "embedding_calls": 1 if mode in ("vector", "local", "global") else 0,
        "tokens_in": len(question) // 2,
        "tokens_out": len(answer_str) // 4,
        "est_cost_usd": 0.0,
    }

    trace = eval_context.build_trace(
        question=question,
        kb=kb,
        mode=mode,
        top_k=top_k,
        level=level,
        retrieved=_retrieved_items_from_chunks(chunks, hits),  # type: ignore[arg-type]
        timing=timing,  # type: ignore[arg-type]
        cost=cost,  # type: ignore[arg-type]
        answer=answer,
        defaults=defaults,  # type: ignore[arg-type]
    )

    if eval_out is not None:
        eval_out.write_text(json.dumps(trace, ensure_ascii=False, indent=2))
    else:
        # Plain print bypasses rich markup/highlighting so the JSON is
        # consumer-clean (one line, no ANSI). Tests parse this directly.
        print(json.dumps(trace, ensure_ascii=False))


def cmd_index(
    path: Path = typer.Argument(..., exists=True, help="File or directory to index."),
    kb: str = typer.Option("default", "--kb", "-k", help="Knowledge base name."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into directories."),
    build_graph: bool = typer.Option(
        False,
        "--build-graph",
        help="Also extract entities/relations and build a knowledge graph (slow — one LLM call per chunk).",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="For each file, delete existing chunks with the same name first. "
             "Use when re-indexing a changed file (defends against duplicate / "
             "stale-content drift). Without this flag, indexing APPENDS and a "
             "warning is shown if conflicts are detected.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable pipeline tracing (per-chunk extraction, dendrogram details, ES indexing stats, ...).",
    ),
) -> None:
    """Parse, chunk, embed and index a file or directory into a knowledge base."""
    validate_kb_name(kb)
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
                result = index_file(
                    fp, kb_name=kb, build_graph=build_graph,
                    replace=replace, progress_cb=cb,
                )
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
    param: list[str] = typer.Option(
        [], "--param",
        help="Override a tunable parameter for this run, e.g. "
             "--param vector_similarity_weight=0.3 (repeatable).",
    ),
    eval_trace: bool = typer.Option(
        False, "--eval-trace",
        help="Emit structured JSON trace (question, retrieved chunks, timing, "
             "cost, params) to stdout or --eval-out.",
    ),
    eval_out: Path = typer.Option(
        None, "--eval-out",
        help="Write --eval-trace JSON to this file (default: stdout).",
    ),
) -> None:
    """Ask a single question. Streams the answer to stdout, then prints citations.

    Retrieval modes (Microsoft-GraphRAG-aligned):
      vector  — Original BM25 + dense (default, fastest)
      local   — Entity-centric multi-source retrieval (4 streams: text units,
                community reports, neighbor entities, relations)
      global  — Map-Reduce over community reports (best for thematic queries)
    """
    validate_kb_name(kb)
    from ragkit import eval_context
    from ragkit.cli import observe
    from ragkit.core.generator import generate
    from ragkit.core.retriever import retrieve

    if debug:
        observe.enable_debug()

    # Install --param overrides into the eval-context store. ValueError from
    # malformed input propagates: Typer will print the message and exit non-zero.
    eval_context.set_overrides(param)

    valid_modes = {"vector", "local", "global"}
    if mode not in valid_modes:
        error(f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(valid_modes))}")
        raise typer.Exit(code=2)

    # Per-call mutable buckets for timing + cost. Always populated when
    # --eval-trace is set; cheap to populate unconditionally.
    timing: dict[str, float] = {
        "embed_ms": 0.0,
        "retrieve_es_ms": 0.0,
        "rerank_ms": 0.0,
        "generate_ms": 0.0,
        "total_ms": 0.0,
    }
    hits = None  # populated only for local/global modes
    t_start = time.monotonic()

    try:
        with observe.measure("retrieve_es_ms", timing):
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
        if eval_trace:
            timing["total_ms"] = (time.monotonic() - t_start) * 1000.0
            _emit_eval_trace(
                question=question, kb=kb, mode=mode, top_k=top_k, level=level,
                chunks=chunks, hits=hits, timing=timing,
                answer=answer, llm_calls=1,
                eval_out=eval_out,
            )
        return

    if not chunks:
        warn("No matching chunks in knowledge base — answer may be generic.")
    else:
        console.print(f"\n[dim]Retrieved {len(chunks)} chunk(s) from kb=[cyan]{kb}[/cyan][/dim]\n")

    # Buffer the generated answer so we can include it in the eval trace.
    answer_parts: list[str] = []
    llm_calls = 0
    with observe.measure("generate_ms", timing):
        for event in generate(question, chunks):
            if event.type == "content":
                answer_parts.append(event.text)
                console.print(event.text, end="", soft_wrap=True, highlight=False)
            elif event.type == "thinking" and show_thinking:
                console.print(event.text, end="", style="dim italic", soft_wrap=True, highlight=False)
            elif event.type == "done":
                llm_calls += 1
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

    if eval_trace:
        timing["total_ms"] = (time.monotonic() - t_start) * 1000.0
        _emit_eval_trace(
            question=question, kb=kb, mode=mode, top_k=top_k, level=level,
            chunks=chunks, hits=hits, timing=timing,
            answer="".join(answer_parts), llm_calls=llm_calls,
            eval_out=eval_out,
        )


def cmd_retrieve(
    question: str = typer.Argument(..., help="Question to retrieve for."),
    kb: str = typer.Option("default", "--kb", "-k", help="Knowledge base name."),
    top_k: int = typer.Option(5, "--top-k", help="Top chunks to retrieve."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show query rewriting trace, ES candidates count, rerank timing, ...",
    ),
    param: list[str] = typer.Option(
        [], "--param",
        help="Override a tunable parameter for this run, e.g. "
             "--param vector_similarity_weight=0.3 (repeatable).",
    ),
    eval_trace: bool = typer.Option(
        False, "--eval-trace",
        help="Emit structured JSON trace (question, retrieved chunks, timing, "
             "cost, params) to stdout or --eval-out.",
    ),
    eval_out: Path = typer.Option(
        None, "--eval-out",
        help="Write --eval-trace JSON to this file (default: stdout).",
    ),
) -> None:
    """Run retrieval only (no LLM call) — useful for tuning."""
    validate_kb_name(kb)
    from ragkit import eval_context
    from ragkit.cli import observe
    from ragkit.core.retriever import retrieve

    if debug:
        observe.enable_debug()

    eval_context.set_overrides(param)

    timing: dict[str, float] = {
        "embed_ms": 0.0,
        "retrieve_es_ms": 0.0,
        "rerank_ms": 0.0,
        "generate_ms": 0.0,
        "total_ms": 0.0,
    }
    with observe.measure("total_ms", timing):
        with observe.measure("retrieve_es_ms", timing):
            chunks = retrieve(question, kb_name=kb, top_k=top_k)

    if not chunks and not eval_trace:
        warn("No matches.")
        return

    if chunks:
        for c in chunks:
            console.rule(
                f"[cyan]#{c.rank}[/cyan] {c.document_name} · "
                f"sim={c.similarity:.3f} (vec={c.vector_similarity:.3f}, term={c.term_similarity:.3f})"
            )
            console.print(c.content)
        console.rule()
    elif eval_trace:
        warn("No matches.")

    if eval_trace:
        _emit_eval_trace(
            question=question, kb=kb, mode="vector", top_k=top_k, level=None,
            chunks=chunks, hits=None, timing=timing,
            answer=None, llm_calls=0,
            eval_out=eval_out,
        )


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
    validate_kb_name(name)
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
    keep_graph: bool = typer.Option(
        False,
        "--keep-graph",
        help="Keep the companion graph (ES {name}_graph index + storage/graphs/"
             "{name}.json). Default: delete the graph too.",
    ),
) -> None:
    """Delete a knowledge base. Irreversible.

    By default this deletes BOTH the chunk index ({name}) AND its companion
    graph layer (ES {name}_graph index + storage/graphs/{name}.json). Pass
    --keep-graph to drop only the chunks and leave the graph in place.
    """
    validate_kb_name(name)
    from ragkit.core.kb_manager import delete_kb
    from ragkit.core._ragflow.rag.utils.es_conn import ESConnection

    if not yes:
        what = "knowledge base" if keep_graph else "knowledge base + graph"
        confirm = typer.confirm(f"Delete {what} '{name}'? This cannot be undone.")
        if not confirm:
            info("Cancelled.")
            return

    deleted = delete_kb(name)

    if not keep_graph:
        # Drop the graph companion ES index — best-effort, may not exist.
        # ISS-022: surface failures at warn level instead of silent except.
        try:
            ESConnection().delete_index(f"{name}_graph")
        except Exception as e:
            warn(f"Could not delete companion graph index '{name}_graph': {e}")

        # Drop the graph JSON file too (this was previously orphaned).
        from ragkit.core.graph.store import open_store
        try:
            open_store(name).clear()
        except Exception as e:
            warn(f"Could not delete graph file for '{name}': {e}")

    if deleted:
        success(f"Deleted '{name}'" + (" (graph preserved)" if keep_graph else " (incl. graph)"))
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
