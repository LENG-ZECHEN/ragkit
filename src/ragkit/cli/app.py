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

from ragkit.cli import commands, graph_cmd
from ragkit.cli.repl import cmd_chat

app = typer.Typer(
    name="rag",
    help="A minimal RAG toolkit (vector + graph retrieval, rerank, LLM answer).",
    no_args_is_help=True,
    add_completion=False,
)

kb_app = typer.Typer(name="kb", help="Knowledge base management.", no_args_is_help=True)
graph_app = typer.Typer(name="graph", help="Knowledge graph management.", no_args_is_help=True)
app.add_typer(kb_app)
app.add_typer(graph_app)


# Top-level commands
app.command("index", help="Parse, embed, index. Add --build-graph to also extract a knowledge graph.")(commands.cmd_index)
app.command("ask", help="Ask a question (modes: vector|local|global|hybrid).")(commands.cmd_ask)
app.command("chat", help="Start an interactive chat REPL.")(cmd_chat)
app.command("retrieve", help="Run retrieval only (no LLM call).")(commands.cmd_retrieve)
app.command("doctor", help="Check ES, API key, dict files.")(commands.cmd_doctor)

# kb sub-commands
kb_app.command("list", help="List knowledge bases.")(commands.cmd_kb_list)
kb_app.command("info", help="Show stats for a knowledge base.")(commands.cmd_kb_info)
kb_app.command("delete", help="Delete a knowledge base.")(commands.cmd_kb_delete)

# graph sub-commands
graph_app.command("build", help="Build a knowledge graph from an indexed KB.")(graph_cmd.cmd_graph_build)
graph_app.command("info", help="Show graph stats for a KB.")(graph_cmd.cmd_graph_info)
graph_app.command("show", help="Show one entity and its neighborhood.")(graph_cmd.cmd_graph_show)
graph_app.command("clear", help="Delete the graph file for a KB.")(graph_cmd.cmd_graph_clear)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
