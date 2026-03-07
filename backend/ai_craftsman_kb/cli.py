"""AI Craftsman KB — command line interface.

All output formatting is delegated to ``cli_output`` so this module stays
thin: parse options, call business logic, pass results to the printer.
"""
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

import click
import httpx

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
from .config.models import AppConfig, LLMGatewayConfig

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

        await _preflight_check(config)

        llm_router = LLMRouter(config)

        # Construct ProcessingPipeline for post-ingest embedding
        from .processing.chunker import Chunker
        from .processing.embedder import Embedder
        from .processing.entity_extractor import EntityExtractor
        from .processing.keyword_extractor import KeywordExtractor
        from .processing.pipeline import ProcessingPipeline
        from .search.vector_store import VectorStore

        pipeline = None
        try:
            embedder = Embedder(config)
            chunker = Chunker(
                chunk_size=config.settings.embedding.chunk_size,
                chunk_overlap=config.settings.embedding.chunk_overlap,
            )
            vector_store = VectorStore(config)
            entity_extractor = EntityExtractor(config, llm_router)

            # Only create keyword extractor if keyword_extraction is configured
            keyword_extractor = None
            if config.settings.llm is not None:
                if isinstance(config.settings.llm, LLMGatewayConfig):
                    if "keyword_extraction" in config.settings.llm.tasks:
                        keyword_extractor = KeywordExtractor(config, llm_router)
                elif config.settings.llm.keyword_extraction is not None:
                    keyword_extractor = KeywordExtractor(config, llm_router)

            pipeline = ProcessingPipeline(
                config=config,
                embedder=embedder,
                chunker=chunker,
                vector_store=vector_store,
                entity_extractor=entity_extractor,
                keyword_extractor=keyword_extractor,
            )
        except Exception as e:
            logger.warning("Could not create processing pipeline: %s", e)

        runner = IngestRunner(config, llm_router, db_path, pipeline=pipeline)

        if source:
            # Check if the requested source is disabled
            if source in config.sources.disabled:
                print_warning(f"Source '{source}' is disabled in sources.yaml")
                return

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
            skipped_sources: list[str] = []
        else:
            from .ingestors.runner import INGESTORS

            disabled = set(config.sources.disabled)
            skipped_sources = []
            reports = []
            with make_ingest_progress() as progress:
                for source_type, ingestor_cls in INGESTORS.items():
                    if source_type in disabled:
                        skipped_sources.append(source_type)
                        continue
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

        if skipped_sources:
            print_warning(f"Skipped disabled sources: {', '.join(skipped_sources)}")
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
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Write output to file instead of stdout",
)
@click.option(
    "--format", "fmt",
    type=click.Choice(["markdown", "json"]),
    default=None,
    help="Output format (default: pretty terminal output)",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    source: tuple[str, ...],
    since: str | None,
    limit: int,
    mode: str,
    output: str | None,
    fmt: str | None,
) -> None:
    """Search indexed content.

    When --format is specified, output is rendered as Markdown or JSON.
    Use --output to write the result to a file instead of stdout.
    """
    config: AppConfig = ctx.obj["config"]

    # Validate the --since date before running async code
    if since is not None:
        try:
            from datetime import date
            date.fromisoformat(since)
        except ValueError:
            print_error(
                f"Invalid date format: {since!r}",
                hint="Use ISO 8601 format, e.g. 2025-01-01",
            )
            return

    async def _run() -> None:
        from .db.sqlite import get_db, init_db
        from .processing.embedder import Embedder
        from .search.hybrid import HybridSearch
        from .search.vector_store import VectorStore

        data_dir = _get_data_dir(config)
        await init_db(data_dir)

        async with get_db(data_dir) as conn:
            embedder = Embedder(config)
            vector_store = VectorStore(config)
            searcher = HybridSearch(config, vector_store, embedder)

            results = await searcher.search(
                conn=conn,
                query=query,
                mode=mode,
                source_types=list(source) if source else None,
                since=since,
                limit=limit,
            )

        if not results:
            if mode in ("semantic", "hybrid"):
                # Check if this might be an empty vector store
                try:
                    qdrant_info = vector_store.get_collection_info()
                    if qdrant_info.get("vectors_count", 0) == 0:
                        console.print(
                            "[yellow]No embeddings found. Run [bold]cr ingest[/bold] first.[/yellow]"
                        )
                        return
                except Exception:
                    pass
            console.print("[yellow]No results found.[/yellow]")
            return

        from .cli_output import print_search_results
        print_search_results(results)

    try:
        asyncio.run(_run())
    except Exception as e:
        print_error(f"Search failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


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
    config: AppConfig = ctx.obj["config"]

    async def _run() -> None:
        from .db.sqlite import get_db, init_db
        from .search.entity_search import EntitySearch

        data_dir = _get_data_dir(config)
        await init_db(data_dir)
        async with get_db(data_dir) as conn:
            entity_search = EntitySearch()
            entity_rows = await entity_search.list_entities(
                conn, entity_type=entity_type, limit=top
            )

        print_entities(entity_rows)

    try:
        asyncio.run(_run())
    except Exception as e:
        print_error(f"Could not list entities: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


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
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Write briefing to a Markdown file instead of printing to terminal",
)
@click.pass_context
def briefing(
    ctx: click.Context,
    topic: str,
    run_radar: bool,
    run_ingest: bool,
    output: str | None,
) -> None:
    """Generate a content briefing on a topic.

    When --output is provided, the briefing is written to the specified file
    as a Markdown document with YAML frontmatter.
    """
    config: AppConfig = ctx.obj["config"]  # noqa: F841
    console.print(
        f"[dim]briefing:[/dim] topic=[bold]{topic!r}[/bold] "
        f"radar={run_radar} ingest={run_ingest}"
    )
    if output:
        console.print(f"[dim]output:[/dim] {output}")
    console.print("[yellow]Briefing generator not yet implemented (task_41).[/yellow]")


# ── Server ────────────────────────────────────────────────────────────────────

@cli.command("server")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Backend port (default from settings.yaml: 8000)",
)
@click.option(
    "--no-dashboard",
    is_flag=True,
    default=False,
    help="Do not serve dashboard static files",
)
@click.option(
    "--with-mcp",
    is_flag=True,
    default=False,
    help="Also start MCP server in same process",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Auto-reload on code changes (development mode)",
)
@click.pass_context
def server(
    ctx: click.Context,
    host: str,
    port: int | None,
    no_dashboard: bool,
    with_mcp: bool,
    reload: bool,
) -> None:
    """Start the FastAPI backend (+ dashboard). Access at http://HOST:PORT"""
    config: AppConfig = ctx.obj["config"]
    backend_port = port or config.settings.server.backend_port

    click.echo(f"Starting AI Craftsman KB server at http://{host}:{backend_port}")
    click.echo(f"Dashboard: http://{host}:{backend_port}/")
    click.echo(f"API docs:  http://{host}:{backend_port}/docs")

    if no_dashboard:
        click.echo("Dashboard static files will NOT be served (--no-dashboard)")

    if reload:
        click.echo("Auto-reload enabled (development mode)")

    if with_mcp:
        click.echo("MCP server startup requested (--with-mcp); starting in same process")

    import uvicorn

    # Pass the import string, not the live object, so --reload can work correctly.
    # When --no-dashboard is requested we set an env var that server.py reads at
    # module import time; however, the cleanest approach for a single-process run
    # is to configure the app factory via an environment variable before uvicorn
    # imports the module.
    import os

    if no_dashboard:
        os.environ["CRAFTSMAN_NO_DASHBOARD"] = "1"
    else:
        os.environ.pop("CRAFTSMAN_NO_DASHBOARD", None)

    uvicorn.run(
        "ai_craftsman_kb.server:app",
        host=host,
        port=backend_port,
        reload=reload,
    )


@cli.command("mcp")
@click.pass_context
def mcp_server(ctx: click.Context) -> None:
    """Start the MCP server (stdio transport for Claude Desktop)."""
    config: AppConfig = ctx.obj["config"]
    try:
        from ai_craftsman_kb.mcp_server import run_mcp_server

        run_mcp_server(config)
    except ImportError:
        click.echo(
            "MCP server module not yet available. "
            "Install mcp extra or wait for task_31 to complete.",
            err=True,
        )
        raise SystemExit(1)


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

        # Retrieve Qdrant vector store info; gracefully degrade if unavailable
        qdrant_info: dict = {}
        try:
            from .search.vector_store import VectorStore
            vector_store = VectorStore(config)
            qdrant_info = vector_store.get_collection_info()
        except Exception as qdrant_err:
            logger.debug("Could not retrieve Qdrant info: %s", qdrant_err)

        print_stats(result, qdrant_info=qdrant_info)
    except Exception as e:
        print_error(f"Error fetching stats: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()


# ── Doctor ────────────────────────────────────────────────────────────────────

# Individual doctor check functions — each returns (status, message) where
# status is 'ok' | 'warn' | 'error' and message is a human-readable summary
# with an actionable fix hint when the status is not 'ok'.



async def _preflight_check(config: AppConfig) -> None:
    """Run pre-flight health checks before ingestion.

    Checks Qdrant, embedding server, and LLM config. Prints warnings
    and asks the user to confirm if any issues are found.
    """
    checks: list[tuple[str, Any]] = [
        ("Qdrant", _check_qdrant(config)),
        ("LLM routing", _check_llm_config(config)),
    ]

    # Check embedding server based on provider
    provider = config.settings.embedding.provider
    if provider == "llamacpp":
        # Get base_url from llamacpp provider config, fallback to default
        pcfg = config.settings.providers.get("llamacpp")
        base_url = pcfg.base_url if pcfg and pcfg.base_url else "http://localhost:9990"
        checks.append(("Embedding server (llamacpp)", _check_connectivity(f"{base_url}/health", "llamacpp")))
    elif provider == "openai":
        checks.append(("Embedding API key (OpenAI)", _check_api_key(config, "openai")))

    warnings = []
    for name, coro in checks:
        try:
            status, message = await coro
        except Exception as exc:
            status, message = "error", str(exc)

        if status == "ok":
            icon = "[green]\u2713[/green]"
        elif status == "warn":
            icon = "[yellow]\u26a0[/yellow]"
            warnings.append(f"{name}: {message}")
        else:
            icon = "[red]\u2717[/red]"
            warnings.append(f"{name}: {message}")

        console.print(f"  {icon} {name:<30s} {message}")

    if warnings:
        if not click.confirm("\n  Continue with ingestion?", default=True):
            raise SystemExit(0)


async def _check_llm_config(config: AppConfig) -> tuple[str, str]:
    """Check that the LLM routing section is present in settings.yaml."""
    if config.settings.llm is None:
        return ("error", "llm section missing from settings.yaml — ingest and search commands will not work")
    tasks = ["filtering", "entity_extraction", "briefing", "source_discovery", "keyword_extraction"]
    if isinstance(config.settings.llm, LLMGatewayConfig):
        configured = [t for t in tasks if t in config.settings.llm.tasks]
        pools = len(config.settings.llm.pools)
        eps = len(config.settings.llm.endpoints)
        return ("ok", f"{len(configured)}/{len(tasks)} tasks configured (gateway: {eps} endpoints, {pools} pools)")
    configured = [t for t in tasks if getattr(config.settings.llm, t, None) is not None]
    return ("ok", f"{len(configured)}/{len(tasks)} tasks configured")


async def _check_database(config: AppConfig) -> tuple[str, str]:
    """Open the SQLite database and count documents.

    Args:
        config: Loaded application configuration (provides data_dir).

    Returns:
        ('ok', doc count message) or ('error', error description with fix hint).
    """
    import aiosqlite

    data_dir = Path(config.settings.data_dir).expanduser()
    db_path = data_dir / "craftsman.db"
    try:
        async with aiosqlite.connect(db_path) as conn:
            async with conn.execute("SELECT COUNT(*) FROM documents") as cursor:
                row = await cursor.fetchone()
                count = row[0] if row else 0
        return ("ok", f"{count} documents")
    except Exception as exc:
        return (
            "error",
            f"Cannot open DB at {db_path}: {exc}. Run `cr ingest` to initialise.",
        )


async def _check_qdrant(config: AppConfig) -> tuple[str, str]:
    """Initialise VectorStore and query the collection info.

    If the local Qdrant storage is locked (another process holds it, e.g. the
    running ``cr server``), treat it as OK — the storage is in use and healthy.

    Args:
        config: Loaded application configuration (provides Qdrant path).

    Returns:
        ('ok', vector count) or ('error', description with fix hint).
    """
    try:
        from .search.vector_store import VectorStore

        vs = VectorStore(config)
        info = vs.get_collection_info()
        count = info.get("vectors_count", 0)
        return ("ok", f"{count} vectors")
    except Exception as exc:
        err_str = str(exc)
        if "already accessed" in err_str:
            return ("ok", "storage locked by running server (healthy)")
        return (
            "error",
            f"Qdrant unavailable: {exc}. Run `cr ingest` to initialise.",
        )


async def _check_api_key(config: AppConfig, provider: str) -> tuple[str, str]:
    """Check that the named LLM provider has an API key configured.

    Missing keys are reported as 'warn' (not 'error') because providers are
    optional — the user may not need every provider.

    Args:
        config: Loaded application configuration.
        provider: Provider name, e.g. 'openai', 'anthropic', 'openrouter'.

    Returns:
        ('ok', masked key prefix) or ('warn', hint to set the env var).
    """
    pcfg = config.settings.providers.get(provider)
    if pcfg and pcfg.api_key:
        masked = pcfg.api_key[:6] + "…"
        return ("ok", f"key set ({masked})")
    if pcfg and pcfg.base_url:
        # Key not needed but note it's a local provider (reachability checked separately)
        return ("ok", f"local provider ({pcfg.base_url})")
    env_var = f"{provider.upper()}_API_KEY"
    return ("warn", f"Not configured — run `direnv allow` or set {env_var} in ~/.ai-craftsman-kb/.env")


async def _check_local_provider(config: AppConfig, provider: str) -> tuple[str, str]:
    """Check that a local LLM provider (ollama, llamacpp) is reachable.

    Args:
        config: Loaded application configuration.
        provider: Provider name ('ollama' or 'llamacpp').

    Returns:
        ('ok', url + status) or ('warn', unreachable hint).
    """
    pcfg = config.settings.providers.get(provider)
    if not pcfg or not pcfg.base_url:
        return ("warn", f"Not configured — add {provider}.base_url to settings.yaml")
    url = pcfg.base_url.rstrip("/")
    # ollama exposes GET / or /api/tags; llamacpp exposes GET /health
    health_path = "/health" if provider == "llamacpp" else "/"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{url}{health_path}")
        return ("ok", f"reachable at {url} (HTTP {resp.status_code})")
    except Exception:
        return ("warn", f"not reachable at {url} — start the {provider} server")


async def _check_youtube_key(config: AppConfig) -> tuple[str, str]:
    """Check that the YouTube Data API key is configured.

    Args:
        config: Loaded application configuration.

    Returns:
        ('ok', masked key) or ('warn', hint to set YOUTUBE_API_KEY).
    """
    yt = config.settings.youtube
    if yt.api_key:
        masked = yt.api_key[:6] + "…"
        return ("ok", f"key set ({masked})")
    return ("warn", "Not configured — run `direnv allow` or set YOUTUBE_API_KEY in ~/.ai-craftsman-kb/.env")


async def _check_reddit_credentials(config: AppConfig) -> tuple[str, str]:
    """Check that Reddit OAuth credentials (client_id + client_secret) are set.

    Args:
        config: Loaded application configuration.

    Returns:
        ('ok', masked id) or ('warn', hint to set REDDIT_CLIENT_ID / SECRET).
    """
    reddit = config.settings.reddit
    if reddit.client_id and reddit.client_secret:
        masked_id = reddit.client_id[:6] + "…"
        return ("ok", f"client_id set ({masked_id})")
    missing = []
    if not reddit.client_id:
        missing.append("REDDIT_CLIENT_ID")
    if not reddit.client_secret:
        missing.append("REDDIT_CLIENT_SECRET")
    return (
        "warn",
        f"Not configured — run `direnv allow` or set {', '.join(missing)} in ~/.ai-craftsman-kb/.env",
    )


async def _check_connectivity(url: str, _name: str) -> tuple[str, str]:
    """Perform a GET request to *url* with a 5-second timeout.

    Args:
        url: The URL to check.
        _name: Human-readable name for the service (unused but kept for call-site clarity).

    Returns:
        ('ok', HTTP status) or ('error', 'Unreachable: <reason>').
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
        return ("ok", f"HTTP {response.status_code}")
    except Exception as exc:
        return ("error", f"Unreachable: {exc}")


async def _check_hn_connectivity() -> tuple[str, str]:
    """Check connectivity to the HN Algolia search API.

    Returns:
        Result of _check_connectivity for the HN endpoint.
    """
    return await _check_connectivity(
        "https://hn.algolia.com/api/v1/search?query=test&hitsPerPage=1",
        "HN Algolia",
    )


async def _check_arxiv_connectivity() -> tuple[str, str]:
    """Check connectivity to the ArXiv API.

    Returns:
        Result of _check_connectivity for the ArXiv endpoint.
    """
    return await _check_connectivity(
        "https://export.arxiv.org/api/query?search_query=all:test&max_results=1",
        "ArXiv",
    )


async def _check_keyword_extraction_config(config: AppConfig) -> tuple[str, str]:
    """Check that the keyword_extraction LLM task is configured.

    Args:
        config: Loaded application configuration.

    Returns:
        ('ok', model info) if configured, ('warn', hint) otherwise.
    """
    if config.settings.llm is None:
        return ("warn", "llm section missing — keyword extraction unavailable")
    if isinstance(config.settings.llm, LLMGatewayConfig):
        if "keyword_extraction" in config.settings.llm.tasks:
            pool_name = config.settings.llm.tasks["keyword_extraction"]
            return ("ok", f"pool={pool_name} (gateway mode)")
        return ("warn", "keyword_extraction not configured in llm.tasks")
    ke = config.settings.llm.keyword_extraction
    if ke is None:
        return ("warn", "keyword_extraction not configured in llm section of settings.yaml")
    return ("ok", f"provider={ke.provider} model={ke.model}")


async def _check_keyword_stats(config: AppConfig) -> tuple[str, str]:
    """Report keyword extraction statistics from the database.

    Counts documents with and without extracted keywords to help the user
    gauge processing coverage.

    Args:
        config: Loaded application configuration.

    Returns:
        ('ok', stats summary) or ('warn', description).
    """
    import aiosqlite

    data_dir = Path(config.settings.data_dir).expanduser()
    db_path = data_dir / "craftsman.db"
    try:
        async with aiosqlite.connect(db_path) as conn:
            # Check if the column exists (handles pre-migration DBs)
            async with conn.execute("PRAGMA table_info(documents)") as cursor:
                columns = {row[1] for row in await cursor.fetchall()}
            if "is_keywords_extracted" not in columns:
                return ("warn", "is_keywords_extracted column missing — run `cr ingest` to migrate")

            async with conn.execute(
                "SELECT COUNT(*) FROM documents WHERE is_keywords_extracted = TRUE"
            ) as cur:
                with_kw = (await cur.fetchone())[0]  # type: ignore[index]
            async with conn.execute(
                "SELECT COUNT(*) FROM documents WHERE is_keywords_extracted = FALSE"
            ) as cur:
                without_kw = (await cur.fetchone())[0]  # type: ignore[index]
            total = with_kw + without_kw
            if total == 0:
                return ("ok", "No documents in database")
            pct = (with_kw / total) * 100 if total else 0
            return ("ok", f"{with_kw}/{total} documents have keywords ({pct:.0f}%)")
    except Exception as exc:
        return ("warn", f"Cannot query keyword stats: {exc}")


async def _check_backend_server(config: AppConfig) -> tuple[str, str]:
    """Check if the backend API server is running."""
    port = config.settings.server.backend_port
    url = f"http://localhost:{port}/api/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
        return ("ok", f"running on port {port} (HTTP {resp.status_code})")
    except Exception:
        return ("warn", f"not reachable on port {port} — run `cr server` to start")


async def _check_frontend_server(config: AppConfig) -> tuple[str, str]:
    """Check if the frontend dev server is running."""
    port = config.settings.server.dashboard_port
    url = f"http://localhost:{port}/"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
        return ("ok", f"running on port {port} (HTTP {resp.status_code})")
    except Exception:
        return ("warn", f"not reachable on port {port} — run `cd dashboard && pnpm dev` to start")


async def _check_data_dir(config: AppConfig) -> tuple[str, str]:
    """Check that the data directory exists (or can be created) and is writable.

    Also reports free disk space.

    Args:
        config: Loaded application configuration.

    Returns:
        ('ok', path + free space) or ('error', description with fix hint).
    """
    data_dir = Path(config.settings.data_dir).expanduser()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # Verify writability by creating and removing a sentinel file
        sentinel = data_dir / ".doctor_write_test"
        sentinel.write_text("ok")
        sentinel.unlink()
        # Report free disk space
        usage = shutil.disk_usage(data_dir)
        free_gb = usage.free / (1024 ** 3)
        return ("ok", f"{data_dir} ({free_gb:.1f} GB free)")
    except Exception as exc:
        return ("error", f"Data dir not writable ({data_dir}): {exc}")


async def _run_doctor(config: AppConfig) -> None:
    """Run all health checks and print a coloured report to the console.

    Checks are always run to completion — a single failure does not abort the
    rest. If any check reports 'error', the process exits with code 1 so that
    CI scripts can detect failures via ``cr doctor || exit 1``.

    Args:
        config: Loaded application configuration.

    Raises:
        SystemExit(1): If one or more checks have status 'error'.
    """
    config_dir = Path(config.config_dir)
    data_dir = Path(config.settings.data_dir).resolve()

    # ── Configuration ─────────────────────────────────────────────────────
    console.print("\n[bold]Configuration[/bold]")
    config_files = [
        ("settings.yaml", "LLM routing, providers, embedding, server settings"),
        ("sources.yaml", "Content source definitions (HN, Reddit, YouTube, etc.)"),
        ("filters.yaml", "Per-source content filtering rules"),
    ]
    console.print(f"  Config dir:  {config_dir}")
    console.print(f"  Data dir:    {data_dir}")
    for filename, description in config_files:
        path = config_dir / filename
        if path.exists():
            size = path.stat().st_size
            console.print(f"  [green]✓[/green] {str(path):<50s} {description} ({size} bytes)")
        else:
            console.print(f"  [yellow]⚠[/yellow] {str(path):<50s} {description} (missing)")

    # ── Collect checks by section ──────────────────────────────────────────
    # Each section is a list of (name, coroutine) tuples.
    local_providers = [
        name for name, pcfg in config.settings.providers.items()
        if pcfg.base_url and not pcfg.api_key
    ]

    key_checks: list[tuple[str, Any]] = [
        *[
            (f"{name.capitalize()} API key", _check_api_key(config, name))
            for name in config.settings.providers
            if name not in local_providers
        ],
        ("YouTube API key", _check_youtube_key(config)),
        ("Reddit credentials", _check_reddit_credentials(config)),
    ]

    service_checks: list[tuple[str, Any]] = [
        ("SQLite DB", _check_database(config)),
        ("Qdrant vectors", _check_qdrant(config)),
        *[
            (f"{name.capitalize()} server", _check_local_provider(config, name))
            for name in local_providers
        ],
        ("Backend server", _check_backend_server(config)),
        ("Frontend server", _check_frontend_server(config)),
        ("HN connectivity", _check_hn_connectivity()),
        ("ArXiv connectivity", _check_arxiv_connectivity()),
        ("Data directory", _check_data_dir(config)),
    ]

    config_checks: list[tuple[str, Any]] = [
        ("LLM routing", _check_llm_config(config)),
        ("Keyword extraction config", _check_keyword_extraction_config(config)),
        ("Keyword extraction stats", _check_keyword_stats(config)),
    ]

    sections: list[tuple[str, list[tuple[str, Any]]]] = [
        ("API Keys", key_checks),
        ("Services", service_checks),
        ("Config", config_checks),
    ]

    # ── Run and display ──────────────────────────────────────────────────
    all_ok = True
    all_results: dict[str, str] = {}  # check name → status
    for section_title, checks in sections:
        # Run all checks in the section, collecting results
        results: list[tuple[str, str, str]] = []
        for name, coro in checks:
            try:
                status, message = await coro
            except Exception as exc:
                status, message = "error", str(exc)
            results.append((name, status, message))
            all_results[name] = status

        # Sort: ok first, then warn, then error
        status_order = {"ok": 0, "warn": 1, "error": 2}
        results.sort(key=lambda r: status_order.get(r[1], 3))

        console.print(f"\n[bold]{section_title}[/bold]")
        for name, status, message in results:
            if status == "ok":
                icon = "[green]✓[/green]"
            elif status == "warn":
                icon = "[yellow]⚠[/yellow]"
            else:
                icon = "[red]✗[/red]"
                all_ok = False
            console.print(f"  {icon} {name:<30s} {message}")

    # ── Quick-start hints for services that are down ─────────────────────
    hints: list[str] = []

    if all_results.get("Llamacpp server") != "ok":
        model = config.settings.embedding.model
        # Resolve model path: check models/ subdir relative to cwd, then bare name
        model_path = Path("models") / model
        if not model_path.exists():
            model_path = Path(model)  # fall back to bare name
        hints.append(
            f"  [yellow]→[/yellow] Start llamacpp embedding server:\n"
            f"    llama-server -m {model_path} --port 9990 --embedding"
        )

    if all_results.get("Qdrant vectors") not in ("ok",):
        hints.append(
            "  [yellow]→[/yellow] Initialise Qdrant storage:\n"
            "    uv run cr ingest --source hn"
        )

    if all_results.get("Ollama server") != "ok":
        hints.append(
            "  [yellow]→[/yellow] Start Ollama:\n"
            "    ollama serve"
        )

    if all_results.get("Backend server") != "ok":
        hints.append(
            "  [yellow]→[/yellow] Start backend server:\n"
            "    uv run cr server"
        )

    if hints:
        console.print("\n[bold]Quick Start[/bold]")
        for hint in hints:
            console.print(hint)

    if all_ok:
        console.print("\n[green]All checks passed. System ready.[/green]")
    else:
        console.print("\n[red]Some checks failed. See messages above.[/red]")
        raise SystemExit(1)


@cli.command("doctor")
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Check system health and configuration."""
    config: AppConfig = ctx.obj["config"]
    asyncio.run(_run_doctor(config))
