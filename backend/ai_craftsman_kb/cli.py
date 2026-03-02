"""AI Craftsman KB — command line interface.

All output formatting is delegated to ``cli_output`` so this module stays
thin: parse options, call business logic, pass results to the printer.
"""
import asyncio
import logging
from pathlib import Path

import click

from .cli_output import (
    console,
    make_ingest_progress,
    print_entities,
    print_error,
    print_ingest_report,
    print_radar_report,
    print_stats,
    print_success,
    print_warning,
)
from .config import load_config
from .config.models import AppConfig

logger = logging.getLogger(__name__)


def _get_data_dir(config: AppConfig) -> Path:
    """Resolve and ensure the data directory exists.

    Args:
        config: The loaded application configuration.

    Returns:
        The resolved Path to the data directory (created if missing).
    """
    data_dir = Path(config.settings.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@click.group()
@click.option(
    "--config-dir",
    type=click.Path(),
    default=None,
    help="Path to config directory (default: ~/.ai-craftsman-kb/)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, config_dir: str | None, verbose: bool) -> None:
    """AI Craftsman KB — local content aggregation and search."""
    ctx.ensure_object(dict)
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    ctx.obj["config"] = load_config(Path(config_dir) if config_dir else None)
    ctx.obj["verbose"] = verbose


# ── Ingest ────────────────────────────────────────────────────────────────────

@cli.command("ingest")
@click.option(
    "--source",
    type=str,
    default=None,
    help="Ingest only this source type (e.g. hn, substack, reddit)",
)
@click.pass_context
def ingest_pro(ctx: click.Context, source: str | None) -> None:
    """Pull latest content from all enabled pro-tier sources."""
    config: AppConfig = ctx.obj["config"]

    async def _run() -> None:
        from .db.sqlite import init_db
        from .ingestors.runner import IngestReport, IngestRunner, get_ingestor
        from .llm.router import LLMRouter

        data_dir = _get_data_dir(config)
        db_path = data_dir / "craftsman.db"

        # Ensure DB schema is up to date before ingestion
        await init_db(data_dir)

        llm_router = LLMRouter(config)
        runner = IngestRunner(config, llm_router, db_path)

        if source:
            try:
                ingestor = get_ingestor(source, config)
            except ValueError as e:
                print_error(str(e), hint="Use one of: hn, substack, reddit, arxiv, youtube, rss, devto")
                return

            reports: list[IngestReport] = []
            with make_ingest_progress() as progress:
                task_id = progress.add_task(f"Ingesting {source}...", total=None)
                report = await runner.run_source(ingestor)
                progress.update(
                    task_id,
                    completed=report.stored,
                    total=report.fetched,
                    description=f"[green]Done {source}: {report.stored} stored",
                )
            reports.append(report)
        else:
            from .ingestors.runner import INGESTORS

            reports = []
            with make_ingest_progress() as progress:
                for source_type, ingestor_cls in INGESTORS.items():
                    task_id = progress.add_task(f"Ingesting {source_type}...", total=None)
                    try:
                        ingestor = ingestor_cls(config)
                        report = await runner.run_source(ingestor)
                    except Exception as e:
                        logger.error("Ingestor %s failed: %s", source_type, e)
                        report = IngestReport(
                            source_type=source_type,
                            errors=[f"init failed: {e}"],
                        )
                    progress.update(
                        task_id,
                        completed=report.stored,
                        total=report.fetched if report.fetched else 0,
                        description=f"[green]Done {source_type}: {report.stored} stored",
                    )
                    reports.append(report)

        print_ingest_report(reports)

    asyncio.run(_run())


# ── Search ────────────────────────────────────────────────────────────────────

@cli.command("search")
@click.argument("query")
@click.option("--source", type=str, multiple=True, help="Filter by source type (repeatable)")
@click.option("--since", type=str, default=None, help="Only results after this date (YYYY-MM-DD)")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "semantic", "keyword"]),
    default="hybrid",
    show_default=True,
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    source: tuple[str, ...],
    since: str | None,
    limit: int,
    mode: str,
) -> None:
    """Search indexed content."""
    config: AppConfig = ctx.obj["config"]  # noqa: F841 — used when search is implemented
    # Search engine not yet implemented — show a placeholder notice
    console.print(
        f"[dim]search:[/dim] query=[bold]{query!r}[/bold] "
        f"mode={mode} limit={limit} sources={list(source) or 'all'}"
    )
    console.print("[yellow]Search not yet implemented (task_21).[/yellow]")


# ── Ingest URL ────────────────────────────────────────────────────────────────

@cli.command("ingest-url")
@click.argument("url")
@click.option("--tag", type=str, multiple=True, help="Tag to apply (repeatable)")
@click.pass_context
def ingest_url(ctx: click.Context, url: str, tag: tuple[str, ...]) -> None:
    """Ingest a single URL into the index."""
    config: AppConfig = ctx.obj["config"]
    tags = list(tag)

    async def _run() -> None:
        from .db.sqlite import init_db
        from .ingestors.runner import IngestRunner

        data_dir = _get_data_dir(config)
        db_path = data_dir / "craftsman.db"
        await init_db(data_dir)

        console.print(f"[dim]Ingesting:[/dim] [bold]{url}[/bold]")
        if tags:
            console.print(f"[dim]Tags:[/dim] {tags}")

        runner = IngestRunner(config, llm_router=None, db_path=db_path)
        try:
            report = await runner.ingest_url(url, tags=tags)
        except Exception as exc:
            print_error(f"Failed to ingest URL: {exc}")
            return

        if report.stored:
            print_success(f"Ingested: {url}")
        elif report.skipped_duplicate:
            print_warning(f"Skipped (duplicate): {url}")
        elif report.errors:
            print_error(f"Failed to ingest {url}: {report.errors[0]}")
        else:
            print_warning(f"No content ingested for: {url}")

    asyncio.run(_run())


# ── Entities ──────────────────────────────────────────────────────────────────

@cli.command("entities")
@click.option("--type", "entity_type", type=str, default=None, help="Filter by entity type")
@click.option("--top", type=int, default=20, show_default=True)
@click.pass_context
def entities(ctx: click.Context, entity_type: str | None, top: int) -> None:
    """List top entities by mention count."""
    config: AppConfig = ctx.obj["config"]  # noqa: F841

    async def _run() -> None:
        try:
            import json as _json

            from .db.models import EntityRow
            from .db.queries import search_entities_fts
            from .db.sqlite import get_db, init_db

            data_dir = _get_data_dir(config)
            await init_db(data_dir)
            async with get_db(data_dir) as conn:
                # When entity_type filter is given, use FTS to narrow down;
                # otherwise list all entities ordered by mention_count.
                if entity_type:
                    entity_rows = await search_entities_fts(conn, entity_type, limit=top)
                else:
                    async with conn.execute(
                        "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                        (top,),
                    ) as cursor:
                        raw_rows = await cursor.fetchall()

                    def _parse(row) -> EntityRow:
                        # aiosqlite.Row supports key access; convert to dict manually
                        d = {k: row[k] for k in row.keys()}
                        if isinstance(d.get("metadata"), str):
                            try:
                                d["metadata"] = _json.loads(d["metadata"])
                            except Exception:
                                d["metadata"] = {}
                        return EntityRow(**d)

                    entity_rows = [_parse(r) for r in raw_rows]

            print_entities(entity_rows)
        except Exception as e:
            print_error(f"Could not list entities: {e}")
            if ctx.obj.get("verbose"):
                console.print_exception()

    asyncio.run(_run())


# ── Radar ─────────────────────────────────────────────────────────────────────

@cli.command("radar")
@click.argument("query")
@click.option("--source", type=str, multiple=True, help="Limit to these source types (repeatable)")
@click.option("--since", type=str, default=None, help="Only results after this date (ISO format)")
@click.option("--limit", type=int, default=10, show_default=True, help="Max results per source")
@click.pass_context
def radar(
    ctx: click.Context,
    query: str,
    source: tuple[str, ...],
    since: str | None,
    limit: int,
) -> None:
    """Search the open web on-demand for a topic.

    Fans out concurrently to all enabled radar sources (HN, Reddit, ArXiv,
    DEV.to, YouTube), deduplicates results by URL, stores new documents with
    origin='radar', and prints a numbered list with document IDs for triage.

    Use ``cr promote <id>`` to add a radar result to the pro tier.
    """
    config: AppConfig = ctx.obj["config"]

    async def _run() -> None:
        from .db.queries import list_documents
        from .db.sqlite import get_db, init_db
        from .ingestors.runner import INGESTORS
        from .radar.engine import RadarEngine

        data_dir = _get_data_dir(config)
        await init_db(data_dir)

        # Instantiate all available ingestors — only those that support
        # search_radar() will return results; others yield an empty list.
        ingestors = {}
        for source_type, ingestor_cls in INGESTORS.items():
            try:
                ingestors[source_type] = ingestor_cls(config)
            except Exception as e:
                logger.debug("Could not instantiate ingestor %s: %s", source_type, e)

        async with get_db(data_dir) as conn:
            engine = RadarEngine(config, ingestors)
            report = await engine.search(
                conn=conn,
                query=query,
                sources=list(source) if source else None,
                limit_per_source=limit,
            )

            # Load the newly ingested radar documents to display them
            radar_docs = await list_documents(
                conn,
                origin="radar",
                limit=report.total_found if report.total_found > 0 else 100,
            )

        # Show results — radar_docs may include older entries; display most recent
        # up to total_found to focus on the current search batch
        displayed_docs = radar_docs[: report.total_found] if report.total_found else []
        duplicate_count = report.total_found - report.new_documents
        print_radar_report(
            results=displayed_docs,
            query=query,
            new_count=report.new_documents,
            duplicate_count=max(0, duplicate_count),
        )

        if report.errors:
            for src, err in report.errors.items():
                print_warning(f"[{src}] {err}")

    try:
        asyncio.run(_run())
    except Exception as e:
        print_error(f"Radar search failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


# ── Triage ────────────────────────────────────────────────────────────────────

@cli.command("promote")
@click.argument("document_id")
@click.pass_context
def promote(ctx: click.Context, document_id: str) -> None:
    """Promote a radar result to pro tier (set promoted_at timestamp).

    After promoting, the document appears alongside pro-tier documents in
    search results. Use the document ID shown by ``cr radar``.
    """
    config: AppConfig = ctx.obj["config"]

    async def _run() -> None:
        from .db.queries import get_document, promote_document
        from .db.sqlite import get_db, init_db

        data_dir = _get_data_dir(config)
        await init_db(data_dir)
        async with get_db(data_dir) as conn:
            doc = await get_document(conn, document_id)
            if doc is None:
                print_error(f"Document not found: {document_id}")
                return
            await promote_document(conn, document_id)
            title = doc.title or "(no title)"
            print_success(f"Promoted: \"{title}\" ({document_id[:8]}...)")

    try:
        asyncio.run(_run())
    except Exception as e:
        print_error(f"Promote failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


@cli.command("archive")
@click.argument("document_id")
@click.pass_context
def archive(ctx: click.Context, document_id: str) -> None:
    """Archive a document (hide from default views).

    Archived documents are excluded from search and list views by default.
    The document is not deleted and can be recovered.
    """
    config: AppConfig = ctx.obj["config"]

    async def _run() -> None:
        from .db.queries import archive_document, get_document
        from .db.sqlite import get_db, init_db

        data_dir = _get_data_dir(config)
        await init_db(data_dir)
        async with get_db(data_dir) as conn:
            doc = await get_document(conn, document_id)
            if doc is None:
                print_error(f"Document not found: {document_id}")
                return
            await archive_document(conn, document_id)
            title = doc.title or "(no title)"
            print_success(f"Archived: \"{title}\" ({document_id[:8]}...)")

    try:
        asyncio.run(_run())
    except Exception as e:
        print_error(f"Archive failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


@cli.command("delete")
@click.argument("document_id")
@click.confirmation_option(prompt="Delete this document permanently?")
@click.pass_context
def delete(ctx: click.Context, document_id: str) -> None:
    """Soft-delete a document (set deleted_at timestamp).

    The document is excluded from all search and list views but is NOT
    physically removed from the database. Pass --yes to skip the confirmation
    prompt (useful in scripts).
    """
    config: AppConfig = ctx.obj["config"]

    async def _run() -> None:
        from .db.queries import get_document, soft_delete_document
        from .db.sqlite import get_db, init_db

        data_dir = _get_data_dir(config)
        await init_db(data_dir)
        async with get_db(data_dir) as conn:
            doc = await get_document(conn, document_id)
            if doc is None:
                print_error(f"Document not found: {document_id}")
                return
            await soft_delete_document(conn, document_id)
            title = doc.title or "(no title)"
            print_success(f"Deleted: \"{title}\" ({document_id[:8]}...)")

    try:
        asyncio.run(_run())
    except Exception as e:
        print_error(f"Delete failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


# ── Briefing ──────────────────────────────────────────────────────────────────

@cli.command("briefing")
@click.argument("topic")
@click.option("--run-radar/--no-radar", default=True, show_default=True)
@click.option("--run-ingest/--no-ingest", default=True, show_default=True)
@click.pass_context
def briefing(ctx: click.Context, topic: str, run_radar: bool, run_ingest: bool) -> None:
    """Generate a content briefing on a topic."""
    config: AppConfig = ctx.obj["config"]  # noqa: F841
    console.print(
        f"[dim]briefing:[/dim] topic=[bold]{topic!r}[/bold] "
        f"radar={run_radar} ingest={run_ingest}"
    )
    console.print("[yellow]Briefing generator not yet implemented (task_41).[/yellow]")


# ── Server ────────────────────────────────────────────────────────────────────

@cli.command("server")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=None, help="Backend port (default from settings.yaml)")
@click.pass_context
def server(ctx: click.Context, host: str, port: int | None) -> None:
    """Start the FastAPI backend + dashboard."""
    config: AppConfig = ctx.obj["config"]
    effective_port = port or config.settings.server.backend_port
    console.print(
        f"[dim]server:[/dim] Starting on [bold]{host}:{effective_port}[/bold]"
    )
    console.print("[yellow]Server command not yet implemented (task_40).[/yellow]")


# ── Stats ─────────────────────────────────────────────────────────────────────

@cli.command("stats")
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show system statistics (documents, entities, sources, briefings)."""
    config: AppConfig = ctx.obj["config"]
    data_dir = _get_data_dir(config)

    async def _stats() -> dict[str, int]:
        from .db.queries import get_stats
        from .db.sqlite import get_db, init_db

        # Ensure schema exists before querying — safe to call on existing DBs
        await init_db(data_dir)
        async with get_db(data_dir) as conn:
            return await get_stats(conn)

    try:
        result = asyncio.run(_stats())
        print_stats(result)
    except Exception as e:
        print_error(f"Error fetching stats: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


# ── Doctor ────────────────────────────────────────────────────────────────────

@cli.command("doctor")
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Check system health: config, API keys, database."""
    config: AppConfig = ctx.obj["config"]
    issues: list[str] = []

    # Check providers — every non-ollama provider needs an API key
    providers = config.settings.providers
    for name, pcfg in providers.items():
        if name != "ollama" and not pcfg.api_key:
            issues.append(f"{name}: API key not set")

    # Resolve data directory
    data_dir = Path(config.settings.data_dir).expanduser()

    from rich.panel import Panel
    from rich.table import Table

    check_table = Table(box=None, show_header=False, pad_edge=False)
    check_table.add_column("Check", style="bold")
    check_table.add_column("Status")

    check_table.add_row("Config loaded", "[green]OK[/green]")
    check_table.add_row(
        "Data directory",
        f"{data_dir} ([green]exists[/green])" if data_dir.exists()
        else f"{data_dir} ([dim]will be created[/dim])",
    )

    console.print(Panel(check_table, title="[bold blue]AI Craftsman KB Doctor[/bold blue]", expand=False))

    if issues:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for issue in issues:
            print_warning(issue)
    else:
        print_success("All checks passed.")
