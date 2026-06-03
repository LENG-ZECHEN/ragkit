"""Rich-based UI helpers shared across CLI commands."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def error(msg: str) -> None:
    console.print(f"[bold red]✗[/bold red] {msg}")


def success(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def info(msg: str) -> None:
    console.print(f"[bold cyan]ℹ[/bold cyan] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow] {msg}")


def panel(title: str, body: str, style: str = "cyan") -> None:
    console.print(Panel(body, title=title, border_style=style))


def kv_table(title: str, items: list[tuple[str, str]]) -> Table:
    """Build a two-column key/value table."""
    table = Table(title=title, show_header=False, border_style="dim")
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value")
    for k, v in items:
        table.add_row(k, v)
    return table
