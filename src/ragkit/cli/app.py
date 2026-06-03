"""ragkit CLI entry point.

Usage:
    rag index <path> [--kb default]
    rag ask "..." [--kb default]
    rag chat [--kb default]
    rag retrieve "..." [--kb default]
    rag kb list | info <name> | delete <name>
    rag doctor
"""

from __future__ import annotations

import typer

from ragkit.cli import commands
from ragkit.cli.repl import cmd_chat

app = typer.Typer(
    name="rag",
    help="A minimal RAG toolkit (hybrid retrieval, rerank, LLM answer).",
    no_args_is_help=True,
    add_completion=False,
)

kb_app = typer.Typer(name="kb", help="Knowledge base management.", no_args_is_help=True)
app.add_typer(kb_app)


# Top-level commands
app.command("index", help="Parse, embed and index files into a knowledge base.")(commands.cmd_index)
app.command("ask", help="Ask a single question against a knowledge base.")(commands.cmd_ask)
app.command("chat", help="Start an interactive chat REPL.")(cmd_chat)
app.command("retrieve", help="Run retrieval only (no LLM call).")(commands.cmd_retrieve)
app.command("doctor", help="Check ES, API key, dict files.")(commands.cmd_doctor)

# kb sub-commands
kb_app.command("list", help="List knowledge bases.")(commands.cmd_kb_list)
kb_app.command("info", help="Show stats for a knowledge base.")(commands.cmd_kb_info)
kb_app.command("delete", help="Delete a knowledge base.")(commands.cmd_kb_delete)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
