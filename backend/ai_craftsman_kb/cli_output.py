"""Rich output helpers for the AI Craftsman KB CLI.

All reusable rich formatting lives here so that cli.py stays thin.
Import the module-level ``console`` for direct printing, or call the
helper functions to render standard output types (tables, panels, etc.).

Note on testing: the module-level ``console`` is intentionally created with
``Console()`` so that it writes to the current ``sys.stdout`` at call time.
Click's test runner patches ``sys.stdout`` to a ``StringIO`` buffer, which
means rich output is captured by the runner when ``force_terminal=False``
(the default in non-TTY environments such as test suites).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .db.models import DocumentRow, EntityRow
    from .ingestors.runner import IngestReport
    from .radar.engine import RadarReport as RadarEngineReport

# ---------------------------------------------------------------------------
# Console singleton — created without an explicit file so that Rich resolves
# sys.stdout lazily at each write call.  This allows Click's CliRunner to
# capture rich output by temporarily replacing sys.stdout during tests.
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# Source-type colour map
# ---------------------------------------------------------------------------

SOURCE_COLORS: dict[str, str] = {
    "hn": "orange3",
    "hackernews": "orange3",
    "arxiv": "blue",
    "youtube": "red",
    "reddit": "orange1",
    "substack": "cyan",
    "rss": "green",
    "devto": "magenta",
    "adhoc": "white",
}

# Entity-type colour map
ENTITY_TYPE_COLORS: dict[str, str] = {
    "person": "cyan",
    "company": "yellow",
    "technology": "green",
    "event": "magenta",
    "book": "blue",
    "paper": "blue",
    "product": "bright_blue",
}


def _source_badge(source_type: str) -> Text:
    """Return a coloured Text badge for the given source_type.

    Args:
        source_type: The source type string (e.g. 'hn', 'arxiv').

    Returns:
        A rich Text object with the appropriate colour applied.
    """
    color = SOURCE_COLORS.get(source_type.lower(), "white")
    return Text(source_type, style=f"bold {color}")


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


def print_search_results(results: list) -> None:
    """Print search results as a rich table with clickable URLs.

    Each row shows: rank, title (with terminal hyperlink), source badge,
    publication date, and relevance score.

    Args:
        results: A list of objects with a ``.document`` attribute
                 (DocumentRow) and a ``.score`` float attribute, as
                 returned by the hybrid search engine.  When search is not
                 yet implemented the function falls back to plain
                 DocumentRow objects.
    """
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False)
    table.add_column("#", width=3, justify="right")
    table.add_column("Title", min_width=30, max_width=60)
    table.add_column("Source", width=12)
    table.add_column("Date", width=12)
    table.add_column("Score", width=7, justify="right")

    for i, result in enumerate(results, 1):
        # Support both SearchResult-like objects and plain DocumentRow objects
        if hasattr(result, "document"):
            doc = result.document
            score = f"{result.score:.2f}"
        else:
            doc = result
            score = "—"

        title_text = Text(doc.title or "(no title)")
        if doc.url:
            title_text.stylize(f"link {doc.url}")

        date_str = (doc.published_at[:10] if doc.published_at else "—")

        table.add_row(
            str(i),
            title_text,
            _source_badge(doc.source_type),
            date_str,
            score,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Entity listing
# ---------------------------------------------------------------------------


def print_entities(entities: list[EntityRow]) -> None:
    """Print entities as a rich colour-coded table grouped by type.

    Args:
        entities: A list of EntityRow objects to display.
    """
    if not entities:
        console.print("[yellow]No entities found.[/yellow]")
        return

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        pad_edge=False,
    )
    table.add_column("Name", min_width=20, max_width=40)
    table.add_column("Type", width=14)
    table.add_column("Mentions", width=10, justify="right")
    table.add_column("First seen", width=12)

    for entity in entities:
        type_color = ENTITY_TYPE_COLORS.get(entity.entity_type.lower(), "white")
        type_text = Text(entity.entity_type, style=f"bold {type_color}")
        date_str = entity.first_seen_at[:10] if entity.first_seen_at else "—"

        table.add_row(
            entity.name,
            type_text,
            str(entity.mention_count),
            date_str,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Stats panel
# ---------------------------------------------------------------------------


def print_stats(stats: dict[str, int], qdrant_info: dict | None = None) -> None:
    """Print a rich panel showing system statistics.

    Displays document counts, entity counts, source counts, and optional
    Qdrant vector store information.

    Args:
        stats: Dict as returned by ``db.queries.get_stats()``.
        qdrant_info: Optional dict with Qdrant vector count and size info.
    """
    total_docs = stats.get("total_documents", 0)
    embedded = stats.get("embedded_documents", 0)
    embed_pct = f"{embedded / total_docs * 100:.0f}%" if total_docs else "—"

    total_entities = stats.get("total_entities", 0)
    total_sources = stats.get("total_sources", 0)
    total_briefings = stats.get("total_briefings", 0)

    vector_count = (qdrant_info or {}).get("vectors_count", "—")
    db_size = (qdrant_info or {}).get("disk_data_size", None)
    size_str = f"{db_size / 1024 / 1024:.0f} MB" if db_size else "—"

    # Build a two-column grid inside a panel
    left_lines = [
        f"[bold]Documents[/bold]   {total_docs:,}",
        f"[bold]Embedded[/bold]    {embed_pct}",
        f"[bold]Vectors[/bold]     {vector_count}",
    ]
    right_lines = [
        f"[bold]Entities[/bold]    {total_entities:,}",
        f"[bold]Sources[/bold]     {total_sources}",
        f"[bold]Briefings[/bold]   {total_briefings}   Size {size_str}",
    ]

    left_col = "\n".join(left_lines)
    right_col = "\n".join(right_lines)

    inner = Columns([left_col, right_col], expand=True, padding=(0, 2))
    panel = Panel(inner, title="[bold blue]AI Craftsman KB[/bold blue]", expand=False)
    console.print(panel)


# ---------------------------------------------------------------------------
# Ingest report summary
# ---------------------------------------------------------------------------


def print_ingest_report(reports: list[IngestReport]) -> None:
    """Print per-source ingest results as a rich table.

    Args:
        reports: A list of IngestReport objects from IngestRunner.
    """
    if not reports:
        console.print("[yellow]No ingest reports to display.[/yellow]")
        return

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        pad_edge=False,
    )
    table.add_column("Source", width=14)
    table.add_column("Fetched", width=9, justify="right")
    table.add_column("Passed", width=9, justify="right")
    table.add_column("Stored", width=9, justify="right")
    table.add_column("Dupes", width=9, justify="right")
    table.add_column("Errors", width=9, justify="right")

    for r in reports:
        err_count = len(r.errors)
        err_str = Text(str(err_count), style="bold red" if err_count else "")
        table.add_row(
            _source_badge(r.source_type),
            str(r.fetched),
            str(r.passed_filter),
            Text(str(r.stored), style="bold green" if r.stored else ""),
            str(r.skipped_duplicate),
            err_str,
        )

    console.print(table)

    # Print individual errors below the table
    for r in reports:
        for err in r.errors:
            console.print(f"  [red]ERROR [{r.source_type}]:[/red] {err}")


# ---------------------------------------------------------------------------
# Radar results
# ---------------------------------------------------------------------------


def print_radar_results(docs: list[DocumentRow], topic: str = "") -> None:
    """Print radar results in rich panels grouped by source type.

    Args:
        docs: A list of DocumentRow objects returned by the radar engine.
        topic: The search topic string, shown in the panel title.
    """
    if not docs:
        console.print("[yellow]No radar results found.[/yellow]")
        return

    # Group by source type
    by_source: dict[str, list[DocumentRow]] = {}
    for doc in docs:
        by_source.setdefault(doc.source_type, []).append(doc)

    title_prefix = f"Radar: [bold]{topic}[/bold] — " if topic else "Radar — "

    for source_type, source_docs in by_source.items():
        color = SOURCE_COLORS.get(source_type.lower(), "white")
        lines: list[str] = []
        for i, doc in enumerate(source_docs, 1):
            title = doc.title or "(no title)"
            date = doc.published_at[:10] if doc.published_at else "—"
            lines.append(f"  {i}. {title}  [dim]{date}[/dim]")
            if doc.url:
                lines.append(f"     [link={doc.url}][dim]{doc.url}[/dim][/link]")

        panel_body = "\n".join(lines)
        panel = Panel(
            panel_body,
            title=f"{title_prefix}[bold {color}]{source_type}[/bold {color}]",
            expand=False,
        )
        console.print(panel)


def print_radar_report(
    results: list,
    query: str,
    new_count: int,
    duplicate_count: int,
) -> None:
    """Print a numbered list of radar search results with triage hints.

    Outputs a numbered list showing source badge, title, URL, date, and
    document ID. A summary line at the bottom shows new vs duplicate counts
    and prompts the user to use ``cr promote <id>`` for triage.

    Each entry follows the format::

        1. [ARXIV] "Title of the document"
           arxiv.org/... · Jan 15, 2025
           Document ID: abc-123

    Args:
        results: A list of DocumentRow objects to display.
        query: The search query string shown in the header.
        new_count: Number of newly ingested documents in this run.
        duplicate_count: Number of documents already in DB.
    """
    if not results:
        console.print("[yellow]No results found for query:[/yellow] " + query)
        return

    # Determine source labels from results
    sources_found: list[str] = sorted({doc.source_type for doc in results})
    sources_str = ", ".join(sources_found)
    console.print(f"\nSearching across: {sources_str}...\n")

    for i, doc in enumerate(results, 1):
        source_type = doc.source_type.upper()
        # Pad source badge to 6 chars for alignment
        badge = f"[{source_type:<5}]"
        title = doc.title or "(no title)"
        url = doc.url or ""

        # Format date + optional metadata (e.g. HN points from metadata)
        date_str = doc.published_at[:10] if doc.published_at else "—"
        points = doc.metadata.get("score") or doc.metadata.get("points")
        points_str = f" · {points} points" if points else ""

        # Extract domain for display
        domain = url.split("/")[2] if url.startswith("http") else url

        console.print(f"  {i}. [{source_type}] \"{title}\"")
        console.print(f"     {domain} · {date_str}{points_str}")
        # Show a short snippet from content if available
        if doc.raw_content:
            snippet = doc.raw_content[:120].replace("\n", " ").strip()
            if len(doc.raw_content) > 120:
                snippet += "..."
            console.print(f"     [dim]{snippet}[/dim]")
        console.print(f"     [dim]Document ID:[/dim] {doc.id}")
        console.print()

    console.print(
        f"Found [bold green]{new_count} new[/bold green], "
        f"[dim]{duplicate_count} duplicates[/dim]. "
        "Run [bold]cr promote <id>[/bold] to add to pro tier."
    )


# ---------------------------------------------------------------------------
# Progress bar factory
# ---------------------------------------------------------------------------


def make_ingest_progress() -> Progress:
    """Return a configured rich Progress object for ingest operations.

    The returned Progress object should be used as a context manager::

        with make_ingest_progress() as progress:
            task_id = progress.add_task('Ingesting hn...', total=None)
            ...

    Returns:
        A Progress instance with spinner, description, bar, count, and
        elapsed-time columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def print_error(message: str, *, hint: str | None = None) -> None:
    """Print a user-facing error message in red.

    Does NOT raise or exit — that is the caller's responsibility.

    Args:
        message: The primary error message to display.
        hint: An optional follow-up hint shown in dim text.
    """
    console.print(f"[bold red]Error:[/bold red] {message}")
    if hint:
        console.print(f"  [dim]{hint}[/dim]")


def print_warning(message: str) -> None:
    """Print a user-facing warning message in yellow.

    Args:
        message: The warning text to display.
    """
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_success(message: str) -> None:
    """Print a success confirmation message in green.

    Args:
        message: The success text to display.
    """
    console.print(f"[bold green]OK:[/bold green] {message}")
