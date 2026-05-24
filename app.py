"""
GitHub Intelligence Agent -- Rich TUI

An interactive terminal interface for the async LangGraph GitHub agent.
Supports query execution, streaming, thread management, history viewing,
and settings inspection -- all powered by `rich`.

Usage:
    uv run python app.py
"""

from __future__ import annotations

# Force UTF-8 for rich console output on Windows (run before any imports)
import os, sys
try:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import asyncio
import sqlite3
from contextlib import closing
from typing import Any

import rich.traceback
from dotenv import load_dotenv
from rich.console import Console

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# Load env before importing agent (agent itself does this, but we need some vars for the menu)
load_dotenv()

from agent import (
    DB_PATH,
    GITHUB_PAT,
    MCP_SERVER_NAME,
    MCP_SERVER_URL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    run_agent,
)

rich.traceback.install(show_locals=True)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_banner() -> Panel:
    """Rich-styled app banner."""
    text = Text()
    text.append("GitHub Intelligence Agent\n", style="bold cyan")
    text.append("Multi-Server MCP + LangGraph + ChatOllama Cloud", style="dim")
    return Panel(text, border_style="cyan", title_align="center")


def _make_menu() -> Panel:
    """Main menu rendered as a rich Panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(justify="right", style="bold green", width=4)
    table.add_column(style="white")
    options = [
        ("1", "Run Query"),
        ("2", "Stream Query (live tokens)"),
        ("3", "Continue Thread"),
        ("4", "View History"),
        ("5", "Settings"),
        ("Q", "Quit"),
    ]
    for num, desc in options:
        table.add_row(f"[{num}]", desc)
    return Panel(table, title="Main Menu", border_style="green")


# ---------------------------------------------------------------------------
# Feature implementations
# ---------------------------------------------------------------------------

def do_run_query(streaming: bool = False) -> None:
    """Prompt for a query and execute the agent (optionally streaming)."""
    console.print(Panel(
        "Enter your natural-language GitHub query.\n"
        "Examples:\n"
        "  - Summarize open PRs for langchain-ai/langchain\n"
        "  - Who are the top contributors to facebook/react this month?\n"
        "  - List critical issues in vercel/next.js",
        title="Query" if not streaming else "Stream Query",
        border_style="magenta",
    ))
    query = Prompt.ask("Query")
    if not query.strip():
        console.print("[yellow]Cancelled -- empty query.[/yellow]")
        return

    thread_id = Prompt.ask("Thread ID (default: tui_default)", default="tui_default")

    if not streaming:
        # Non-streaming: show spinner while we wait
        with console.status("[bold green]Thinking...", spinner="dots") as status:
            try:
                result = asyncio.run(run_agent(query, thread_id=thread_id))
            except Exception as exc:
                console.print(Panel(str(exc), title="[red]Error[/red]", border_style="red"))
                return
        console.print(Panel(result, title="[bold green]Result[/bold green]", border_style="green"))
    else:
        # Streaming: use Live display
        _stream_result(query, thread_id)


def _stream_result(query: str, thread_id: str) -> None:
    """Run agent with stream=True and update a Live display panel."""
    live_text = Text("", style="white")
    panel = Panel(live_text, title="[bold cyan]Streaming...[/bold cyan]", border_style="cyan")

    async def _inner() -> str:
        return await run_agent(query, thread_id=thread_id, stream=True)

    with Live(panel, console=console, refresh_per_second=10) as live:
        # Start spinner while waiting for the first token
        live_text.append("", style="dim")
        try:
            result = asyncio.run(_inner())
        except Exception as exc:
            live.update(Panel(str(exc), title="[red]Error[/red]", border_style="red"))
            return
        live.update(Panel(result, title="[bold green]Result[/bold green]", border_style="green"))


def do_continue_thread() -> None:
    """List existing threads and let the user continue one."""
    threads = _list_threads()
    if not threads:
        console.print("[yellow]No conversation threads found.[/yellow]")
        return

    table = Table(title="Conversation Threads", border_style="blue")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Thread ID")
    table.add_column("Messages", style="dim")

    for idx, (tid, count) in enumerate(threads, start=1):
        table.add_row(str(idx), tid, str(count))
    console.print(table)

    choice = Prompt.ask("Select thread number (or 'c' to cancel)", default="c")
    if choice.lower() == "c":
        return
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(threads):
        console.print("[red]Invalid selection.[/red]")
        return

    selected_tid = threads[int(choice) - 1][0]
    query = Prompt.ask(f"New message for thread [bold]{selected_tid}[/bold]")
    if not query.strip():
        console.print("[yellow]Cancelled.[/yellow]")
        return

    with console.status(f"[bold green]Continuing thread {selected_tid}...", spinner="dots") as status:
        try:
            result = asyncio.run(run_agent(query, thread_id=selected_tid))
        except Exception as exc:
            console.print(Panel(str(exc), title="[red]Error[/red]", border_style="red"))
            return
    console.print(Panel(result, title="[bold green]Result[/bold green]", border_style="green"))


def do_view_history() -> None:
    """Display past conversation summaries from the SQLite checkpoint DB."""
    threads = _list_threads()
    if not threads:
        console.print("[yellow]No history found.[/yellow]")
        return

    table = Table(title="History Overview", border_style="blue")
    table.add_column("Thread ID", style="bold")
    table.add_column("Messages", justify="right")
    table.add_column("DB Path", style="dim")

    for tid, count in threads:
        table.add_row(tid, str(count), DB_PATH)

    console.print(table)
    console.print(f"\n[dim]Full checkpoint DB: {DB_PATH}[/dim]")


def do_settings() -> None:
    """Show current environment / configuration."""
    table = Table(title="Current Settings", border_style="yellow")
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    table.add_row("GITHUB_PAT", _mask(GITHUB_PAT), "env")
    table.add_row("OLLAMA_API_KEY", _mask(OLLAMA_API_KEY), "env")
    table.add_row("OLLAMA_BASE_URL", OLLAMA_BASE_URL, "env")
    table.add_row("OLLAMA_MODEL", OLLAMA_MODEL, "env")
    table.add_row("DB_PATH", DB_PATH, "env")
    table.add_row("MCP_SERVER", f"{MCP_SERVER_NAME} --> {MCP_SERVER_URL}", "code")
    table.add_row("SYSTEM_PROMPT", SYSTEM_PROMPT[:60] + "...", "code")

    console.print(table)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _list_threads() -> list[tuple[str, int]]:
    """Return [(thread_id, message_count), ...] from the SQLite checkpointer."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
            )
            return [(row[0], row[1]) for row in cur.fetchall()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _mask(value: str, visible: int = 6) -> str:
    """Mask a sensitive string except for the first/last *visible* chars."""
    if len(value) <= visible * 2:
        return "***"
    return f"{value[:visible]}...{value[-visible:]}"


def _clear_screen() -> None:
    """Cross-platform terminal clear."""
    console.clear()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point -- interactive TUI loop."""
    while True:
        _clear_screen()
        console.print(_make_banner())
        console.print(_make_menu())
        console.print()

        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "q", "Q"], default="Q")
        choice = choice.strip().lower()

        if choice == "1":
            do_run_query(streaming=False)
        elif choice == "2":
            do_run_query(streaming=True)
        elif choice == "3":
            do_continue_thread()
        elif choice == "4":
            do_view_history()
        elif choice == "5":
            do_settings()
        elif choice in ("q", "Q"):
            console.print("[bold cyan]Goodbye![/bold cyan]")
            sys.exit(0)
        else:
            console.print("[red]Invalid option.[/red]")

        console.print()
        Prompt.ask("Press Enter to return to menu")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Interrupted. Goodbye![/bold cyan]")
        sys.exit(0)
