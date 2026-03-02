"""AI Craftsman KB — command line interface."""
import asyncio
import logging
from pathlib import Path

import click

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
        from .db.sqlite import init_db, get_db  # noqa: F401
        from .llm.router import LLMRouter
        from .ingestors.runner import IngestRunner, get_ingestor

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
                click.echo(f"Error: {e}", err=True)
                return
            reports = [await runner.run_source(ingestor)]
        else:
            reports = await runner.run_all()

        for r in reports:
            err_count = len(r.errors)
            click.echo(
                f"{r.source_type}: fetched={r.fetched} "
                f"passed={r.passed_filter} stored={r.stored} "
                f"dupes={r.skipped_duplicate} errors={err_count}"
            )
            for err in r.errors:
                click.echo(f"  ERROR: {err}", err=True)

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
    config: AppConfig = ctx.obj["config"]
    click.echo(f"[search] query={query!r} mode={mode} limit={limit} — not yet implemented")


# ── Ingest URL ────────────────────────────────────────────────────────────────

@cli.command("ingest-url")
@click.argument("url")
@click.option("--tag", type=str, multiple=True, help="Tag to apply (repeatable)")
@click.pass_context
def ingest_url(ctx: click.Context, url: str, tag: tuple[str, ...]) -> None:
    """Ingest a single URL into the index."""
    config: AppConfig = ctx.obj["config"]
    click.echo(f"[ingest-url] url={url!r} tags={list(tag)} — not yet implemented")


# ── Entities ──────────────────────────────────────────────────────────────────

@cli.command("entities")
@click.option("--type", "entity_type", type=str, default=None, help="Filter by entity type")
@click.option("--top", type=int, default=20, show_default=True)
@click.pass_context
def entities(ctx: click.Context, entity_type: str | None, top: int) -> None:
    """List top entities by mention count."""
    config: AppConfig = ctx.obj["config"]
    click.echo(f"[entities] type={entity_type} top={top} — not yet implemented")


# ── Radar ─────────────────────────────────────────────────────────────────────

@cli.command("radar")
@click.argument("query")
@click.option("--source", type=str, multiple=True, help="Limit to these source types (repeatable)")
@click.option("--since", type=str, default=None)
@click.pass_context
def radar(ctx: click.Context, query: str, source: tuple[str, ...], since: str | None) -> None:
    """Search the open web on-demand for a topic."""
    config: AppConfig = ctx.obj["config"]
    click.echo(f"[radar] query={query!r} sources={list(source)} — not yet implemented")


# ── Briefing ──────────────────────────────────────────────────────────────────

@cli.command("briefing")
@click.argument("topic")
@click.option("--run-radar/--no-radar", default=True, show_default=True)
@click.option("--run-ingest/--no-ingest", default=True, show_default=True)
@click.pass_context
def briefing(ctx: click.Context, topic: str, run_radar: bool, run_ingest: bool) -> None:
    """Generate a content briefing on a topic."""
    config: AppConfig = ctx.obj["config"]
    click.echo(f"[briefing] topic={topic!r} radar={run_radar} ingest={run_ingest} — not yet implemented")


# ── Server ────────────────────────────────────────────────────────────────────

@cli.command("server")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=None, help="Backend port (default from settings.yaml)")
@click.pass_context
def server(ctx: click.Context, host: str, port: int | None) -> None:
    """Start the FastAPI backend + dashboard."""
    config: AppConfig = ctx.obj["config"]
    effective_port = port or config.settings.server.backend_port
    click.echo(f"[server] Starting on {host}:{effective_port} — not yet implemented")


# ── Stats ─────────────────────────────────────────────────────────────────────

@cli.command("stats")
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show system statistics (documents, entities, sources, briefings)."""
    config: AppConfig = ctx.obj["config"]
    data_dir = _get_data_dir(config)

    async def _stats() -> dict[str, int]:
        from .db.sqlite import get_db, init_db
        from .db.queries import get_stats

        # Ensure schema exists before querying — safe to call on existing DBs
        await init_db(data_dir)
        async with get_db(data_dir) as conn:
            return await get_stats(conn)

    try:
        result = asyncio.run(_stats())
        click.echo("=== AI Craftsman KB Stats ===")
        for key, val in result.items():
            click.echo(f"  {key}: {val}")
    except Exception as e:
        click.echo(f"Error fetching stats: {e}", err=True)


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
            issues.append(f"  - {name}: API key not set")

    # Resolve data directory
    data_dir = Path(config.settings.data_dir).expanduser()

    click.echo("=== AI Craftsman KB Doctor ===")
    click.echo("Config loaded: OK")
    click.echo(f"Data dir: {data_dir} ({'exists' if data_dir.exists() else 'will be created'})")

    if issues:
        click.echo("\nWarnings:")
        for issue in issues:
            click.echo(issue)
    else:
        click.echo("All checks passed.")
