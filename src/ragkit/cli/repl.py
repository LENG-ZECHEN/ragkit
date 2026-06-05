"""Interactive REPL for ragkit chat.

Phase A goal (task #27): the REPL exposes every retrieval capability the
single-shot ``rag ask`` command has — modes, debug tracing, level filter
— while keeping each turn semantically independent (no conversation
history yet; that's phase B / task #28).

State machine summary
---------------------
ReplState carries everything per session: knowledge base, retrieval mode,
top_k, debug toggle, optional global-level filter, plus a cache of the
most recent retrieval (so ``/show N`` works).

Slash commands mutate (a copy of) the state. Free text is treated as a
question and routed to the right retriever via state.mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.table import Table

from ragkit.cli import observe
from ragkit.cli.ui import console, error, info, warn
from ragkit.core.kb_validator import validate_kb_name
from ragkit.core.retriever import RetrievedChunk

PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})

VALID_MODES = ("vector", "local", "global")


@dataclass
class ReplState:
    """Mutable session state for the REPL."""

    kb: str
    top_k: int
    show_thinking: bool
    last_chunks: list[RetrievedChunk]
    # ---- phase A additions: align with `rag ask` capabilities ----
    mode: str = "vector"             # vector | local | global
    debug: bool = False              # mirrored to observe module
    level: int | None = None         # global mode hierarchy filter (None = cross-level)


HELP_TEXT = """\
[bold]REPL commands[/bold]
  [cyan]/kb <name>[/cyan]             switch knowledge base
  [cyan]/mode <vector|local|global>[/cyan]   set retrieval mode
  [cyan]/level <N>[/cyan]             (global only) restrict to community level N; "/level none" clears
  [cyan]/top <n>[/cyan]               set top_k for retrieval (1-20)
  [cyan]/thinking[/cyan]              toggle LLM reasoning display
  [cyan]/debug[/cyan]                 toggle pipeline trace (vector / local / global internals)
  [cyan]/show <i>[/cyan]              print the full text of reference i (after a question)
  [cyan]/clear[/cyan]                 clear the screen
  [cyan]/help[/cyan]                  show this help
  [cyan]/exit[/cyan]                  leave the REPL
Anything else is treated as a question.
"""


# --------------------------------------------------------------------------
# Slash-command handlers
# --------------------------------------------------------------------------


def _handle_command(line: str, state: ReplState) -> ReplState | None:
    """Process a /command. Returns updated state, or None to exit.

    Each branch returns a (possibly new) ReplState — callers replace the
    loop's state with the returned object so failed/no-op commands don't
    silently mutate.
    """
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in {"/exit", "/quit", "/q"}:
        return None

    if cmd == "/help":
        console.print(HELP_TEXT)
        return state

    if cmd == "/clear":
        console.clear()
        return state

    if cmd == "/kb":
        if not arg:
            error("Usage: /kb <name>")
            return state
        info(f"Switched to kb=[cyan]{arg}[/cyan]")
        return replace(state, kb=arg)

    if cmd == "/mode":
        if arg not in VALID_MODES:
            error(f"Usage: /mode <{ '|'.join(VALID_MODES) }>")
            return state
        # When leaving global, clear the level filter (it's meaningless elsewhere).
        new_level = state.level if arg == "global" else None
        info(f"Mode set to [cyan]{arg}[/cyan]")
        return replace(state, mode=arg, level=new_level)

    if cmd == "/level":
        # Only meaningful for global mode. Allow setting in other modes
        # (with a warning) since the user may set it then switch.
        if not arg or arg.lower() in {"none", "0", "clear"}:
            info("Level filter cleared (cross-level vector search)")
            return replace(state, level=None)
        try:
            n = int(arg)
            if n < 0:
                raise ValueError
        except ValueError:
            error("Usage: /level <non-negative integer> or /level none")
            return state
        if state.mode != "global":
            warn(f"Level only affects global mode (current: {state.mode})")
        info(f"Level set to [cyan]{n}[/cyan]")
        return replace(state, level=n)

    if cmd == "/top":
        try:
            n = int(arg)
            if not 1 <= n <= 20:
                raise ValueError
        except ValueError:
            error("Usage: /top <integer 1-20>")
            return state
        info(f"top_k set to {n}")
        return replace(state, top_k=n)

    if cmd == "/thinking":
        new_state = replace(state, show_thinking=not state.show_thinking)
        info(f"Thinking display: {'on' if new_state.show_thinking else 'off'}")
        return new_state

    if cmd == "/debug":
        new_debug = not state.debug
        # Keep the observe module's global flag in sync so any trace_xxx
        # call inside the pipeline lights up immediately.
        if new_debug:
            observe.enable_debug()
        else:
            observe.disable_debug()
        info(f"Debug trace: {'on' if new_debug else 'off'}")
        return replace(state, debug=new_debug)

    if cmd == "/show":
        if not state.last_chunks:
            warn("No references to show — ask a question first.")
            return state
        try:
            idx = int(arg) - 1
            if not 0 <= idx < len(state.last_chunks):
                raise ValueError
        except ValueError:
            error(f"Usage: /show <1-{len(state.last_chunks)}>")
            return state
        c = state.last_chunks[idx]
        console.rule(f"[cyan]#{c.rank}[/cyan] {c.document_name} · sim={c.similarity:.3f}")
        console.print(c.content)
        console.rule()
        return state

    error(f"Unknown command: {cmd}. Type /help for help.")
    return state


# --------------------------------------------------------------------------
# Mode dispatch for question handling
# --------------------------------------------------------------------------


def _retrieve_for_mode(question: str, state: ReplState):
    """Route the question to the right retriever based on state.mode.

    Returns ``(chunks, raw_hits_or_None)``:
      - chunks: list[RetrievedChunk] (always non-None; what the generator consumes)
      - raw_hits: list[GraphHit] for non-vector modes (used for kind-aware
        References table); None for vector mode.

    TODO(task #28): factor out this dispatch — cmd_ask in commands.py has
    a near-identical block. The two are intentionally duplicated for now
    so phase A doesn't pre-judge phase B's needs.
    """
    from ragkit.core.retriever import retrieve

    if state.mode == "vector":
        return retrieve(question, kb_name=state.kb, top_k=state.top_k), None

    from ragkit.core.graph.retriever import (
        graph_hits_to_chunks,
        retrieve_global,
        retrieve_local,
    )

    if state.mode == "local":
        hits = retrieve_local(question, kb_name=state.kb, top_k=state.top_k)
    else:  # global
        hits = retrieve_global(
            question, kb_name=state.kb, top_k=state.top_k, level=state.level
        )
    return graph_hits_to_chunks(hits), hits


def _ask(question: str, state: ReplState) -> ReplState:
    """One retrieval + generation cycle. Honors state.mode / debug / level."""
    from ragkit.core.generator import generate

    try:
        chunks, hits = _retrieve_for_mode(question, state)
    except Exception as e:
        error(f"Retrieval failed ({state.mode}): {e}")
        return state

    if not chunks:
        warn("No matching chunks — answer may be generic.")
    else:
        console.print(
            f"\n[dim]Retrieved {len(chunks)} chunk(s) from kb=[cyan]{state.kb}[/cyan] "
            f"(mode=[cyan]{state.mode}[/cyan])[/dim]\n"
        )

    for event in generate(question, chunks):
        if event.type == "content":
            console.print(event.text, end="", soft_wrap=True, highlight=False)
        elif event.type == "thinking" and state.show_thinking:
            console.print(event.text, end="", style="dim italic", soft_wrap=True, highlight=False)
        elif event.type == "error":
            console.print()
            error(f"Generation error: {event.text}")
            return state

    console.print()

    # References rendering — kind-aware for non-vector modes.
    if chunks:
        if state.mode == "vector":
            table = Table(show_header=True, border_style="dim")
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Document")
            table.add_column("Sim", justify="right")
            for c in chunks:
                table.add_row(str(c.rank), c.document_name, f"{c.similarity:.3f}")
            console.print(table)
        else:
            # hits is the GraphHit list (carries .kind for the new column)
            console.print(observe.references_table_with_kind(hits or []))

    return replace(state, last_chunks=list(chunks))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def cmd_chat(
    kb: str = typer.Option("default", "--kb", "-k", help="Knowledge base name."),
    top_k: int = typer.Option(5, "--top-k", help="Top chunks per question."),
    mode: str = typer.Option(
        "vector",
        "--mode",
        "-m",
        help="Initial retrieval mode: vector | local | global.",
    ),
    level: int = typer.Option(
        None,
        "--level",
        help="(global only) Initial community level filter.",
    ),
    thinking: bool = typer.Option(False, "--thinking", help="Show LLM reasoning trace."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable pipeline trace output (toggle later with /debug).",
    ),
) -> None:
    """Start an interactive REPL. Type /help inside for commands."""
    validate_kb_name(kb)
    if mode not in VALID_MODES:
        error(f"Invalid --mode '{mode}'. Choose from: {', '.join(VALID_MODES)}")
        raise typer.Exit(code=2)

    # Sync observe with the initial --debug; subsequent /debug toggles
    # in-session will keep them aligned.
    if debug:
        observe.enable_debug()
    else:
        observe.disable_debug()

    history_path = Path.home() / ".rag" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(history_path)))

    state = ReplState(
        kb=kb,
        top_k=top_k,
        show_thinking=thinking,
        last_chunks=[],
        mode=mode,
        debug=debug,
        level=level,
    )

    # Startup banner: surface every per-session setting.
    banner_parts = [
        f"kb=[cyan]{state.kb}[/cyan]",
        f"mode=[cyan]{state.mode}[/cyan]",
        f"top_k={state.top_k}",
    ]
    if state.level is not None:
        banner_parts.append(f"level={state.level}")
    if state.debug:
        banner_parts.append("[yellow]debug[/yellow]")
    console.print(
        f"[bold]ragkit REPL[/bold] · "
        + " · ".join(banner_parts)
        + " · type [cyan]/help[/cyan] for commands\n"
    )

    while True:
        try:
            line = session.prompt("> ", style=PROMPT_STYLE).strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        if not line:
            continue

        if line.startswith("/"):
            updated = _handle_command(line, state)
            if updated is None:
                break
            state = updated
            continue

        state = _ask(line, state)

    info("Bye.")
