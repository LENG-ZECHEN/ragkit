"""Interactive REPL for ragkit chat."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.table import Table

from ragkit.cli.ui import console, error, info, warn
from ragkit.core.retriever import RetrievedChunk

PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})


@dataclass
class ReplState:
    """Mutable session state for the REPL."""

    kb: str
    top_k: int
    show_thinking: bool
    last_chunks: list[RetrievedChunk]


HELP_TEXT = """\
[bold]REPL commands[/bold]
  [cyan]/kb <name>[/cyan]     switch knowledge base
  [cyan]/top <n>[/cyan]       set top_k for retrieval (1-20)
  [cyan]/thinking[/cyan]      toggle thinking-trace display
  [cyan]/show <i>[/cyan]      print the full text of reference i (after a question)
  [cyan]/clear[/cyan]         clear the screen
  [cyan]/help[/cyan]          show this help
  [cyan]/exit[/cyan]          leave the REPL
Anything else is treated as a question.
"""


def _handle_command(line: str, state: ReplState) -> ReplState | None:
    """Process a /command. Returns updated state, or None to exit."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in {"/exit", "/quit", "/q"}:
        return None

    if cmd == "/help":
        console.print(HELP_TEXT)
        return state

    if cmd == "/clear":
        os.system("clear" if os.name != "nt" else "cls")
        return state

    if cmd == "/kb":
        if not arg:
            error("Usage: /kb <name>")
            return state
        info(f"Switched to kb=[cyan]{arg}[/cyan]")
        return replace(state, kb=arg)

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


def _ask(question: str, state: ReplState) -> ReplState:
    """Run one retrieval + generation cycle."""
    from ragkit.core.generator import generate
    from ragkit.core.retriever import retrieve

    try:
        chunks = retrieve(question, kb_name=state.kb, top_k=state.top_k)
    except Exception as e:
        error(f"Retrieval failed: {e}")
        return state

    if not chunks:
        warn("No matching chunks — answer may be generic.")
    else:
        console.print(f"\n[dim]Retrieved {len(chunks)} chunk(s) from kb=[cyan]{state.kb}[/cyan][/dim]\n")

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
    if chunks:
        table = Table(show_header=True, border_style="dim")
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Document")
        table.add_column("Sim", justify="right")
        for c in chunks:
            table.add_row(str(c.rank), c.document_name, f"{c.similarity:.3f}")
        console.print(table)

    return replace(state, last_chunks=list(chunks))


def cmd_chat(
    kb: str = typer.Option("default", "--kb", "-k", help="Knowledge base name."),
    top_k: int = typer.Option(5, "--top-k", help="Top chunks per question."),
    thinking: bool = typer.Option(False, "--thinking", help="Show LLM reasoning trace."),
) -> None:
    """Start an interactive REPL. Type /help inside for commands."""
    history_path = Path.home() / ".rag" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    session = PromptSession(history=FileHistory(str(history_path)))
    state = ReplState(kb=kb, top_k=top_k, show_thinking=thinking, last_chunks=[])

    console.print(
        f"[bold]ragkit REPL[/bold] · kb=[cyan]{state.kb}[/cyan] · "
        f"top_k={state.top_k} · type [cyan]/help[/cyan] for commands\n"
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
