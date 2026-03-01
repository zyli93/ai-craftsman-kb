# Task 43: Rich CLI Output + Progress Bars

## Wave
Wave 16 (parallel with tasks 42, 44, 45; depends on task 08)
Domain: backend

## Objective
Upgrade all CLI commands from plain `click.echo()` to rich-formatted output using the `rich` library: colored tables, progress bars during ingest, spinners, and formatted panels for search results.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/cli.py` — Replace all `click.echo()` with `rich` equivalents
- `backend/ai_craftsman_kb/cli_output.py` — Reusable rich output helpers

### Key interfaces / implementation details:

**Rich output helpers** (`cli_output.py`):
```python
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

def print_search_results(results: list[SearchResult]) -> None:
    """Print search results as a rich table with clickable URLs."""
    table = Table(box=box.SIMPLE, show_header=True, header_style='bold cyan')
    table.add_column('#', width=3)
    table.add_column('Title', min_width=30, max_width=60)
    table.add_column('Source', width=12)
    table.add_column('Date', width=12)
    table.add_column('Score', width=7)
    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            Text(r.document.title or '(no title)', style='link ' + r.document.url),
            r.document.source_type,
            r.document.published_at[:10] if r.document.published_at else '—',
            f'{r.score:.2f}',
        )
    console.print(table)

def print_entities(entities: list[EntityRow]) -> None:
    """Rich table for entity listing."""

def print_stats(stats: dict, qdrant_info: dict) -> None:
    """Rich panel with stats grid."""

def print_ingest_report(reports: list[IngestReport]) -> None:
    """Rich table showing per-source ingest results."""

def print_radar_results(docs: list[DocumentRow], report: RadarReport) -> None:
    """Rich panels for radar results, grouped by source."""

def make_ingest_progress() -> Progress:
    """Return a configured rich Progress object for ingest operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        TextColumn('{task.completed}/{task.total}'),
        TimeElapsedColumn(),
    )
```

**Ingest progress** during `cr ingest`:
```python
async def _run_ingest(...) -> None:
    with make_ingest_progress() as progress:
        for source_type, ingestor in ingestors.items():
            task_id = progress.add_task(f'Ingesting {source_type}...', total=None)
            report = await runner.run_source(ingestor)
            progress.update(task_id, completed=report.stored, total=report.fetched,
                           description=f'[green]✓ {source_type}: {report.stored} stored')
```

**Search output** — before (plain):
```
1. Title of article
   hn · author · 2025-01-15
   Score: 0.943
```
After (rich):
```
  # │ Title                          │ Source  │ Date       │ Score
  ──┼────────────────────────────────┼─────────┼────────────┼──────
  1 │ Title of article (clickable)   │ hn      │ 2025-01-15 │ 0.94
```

**Stats output** — rich Panel with nested grid:
```
╭─ AI Craftsman KB ────────────────────╮
│  Documents   2,847  │  Entities  14,203 │
│  Embedded    100%   │  Sources     12   │
│  Vectors    50,341  │  Size     623 MB  │
╰───────────────────────────────────────╯
```

**Error output**: Use `console.print_exception()` for unhandled exceptions in debug mode; user-facing errors use `console.print('[red]Error: ...[/red]')`.

**Color scheme**:
- Source type badges: `hn`=orange, `arxiv`=blue, `youtube`=red, `reddit`=orange, `substack`=cyan
- Status: success=green, warning=yellow, error=red

## Dependencies
- Depends on: task_08 (CLI commands to upgrade)
- Packages needed: `rich` (already in pyproject.toml)

## Acceptance Criteria
- [ ] `cr ingest` shows per-source progress bars during fetch
- [ ] `cr search "query"` outputs a formatted table with clickable URLs (terminal hyperlinks)
- [ ] `cr stats` shows a styled panel with all metrics
- [ ] `cr entities` shows a color-coded table grouped by type
- [ ] `cr radar "query"` shows results in rich panels grouped by source
- [ ] Errors shown in red with clear message (no raw Python tracebacks for user-facing errors)
- [ ] `--no-color` or `NO_COLOR` env var disables rich formatting (rich handles this automatically)
- [ ] All output helpers live in `cli_output.py` — CLI stays thin

## Notes
- `rich` `Text` with `style='link URL'` creates clickable terminal hyperlinks in supported terminals (iTerm2, modern terminals)
- Use `console = Console(stderr=False)` for stdout output; progress bars use stderr by default in rich
- `rich.progress.Progress` as async context manager: `async with Progress(...) as progress:`
- `NO_COLOR` environment variable: `rich` respects it automatically — no manual check needed
- Source type colors: define a `SOURCE_COLORS` dict once in `cli_output.py` and use throughout
