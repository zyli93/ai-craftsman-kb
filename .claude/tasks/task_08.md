# Task 08: CLI Skeleton (Click)

## Wave
Wave 3 (parallel with tasks: 06, 07)
Domain: backend

## Objective
Build the Click CLI skeleton with all top-level command groups and placeholder implementations, establishing the `cr` entry point and the config/DB initialization flow used by every command.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/cli.py` — Main Click app with all command groups
- `pyproject.toml` — Add `[project.scripts]` entry: `cr = "ai_craftsman_kb.cli:cli"`
- `backend/tests/test_cli.py` — CLI invocation tests via Click's `CliRunner`

### Key interfaces / implementation details:

**CLI structure** (`cli.py`) — all commands from plan.md:
```python
import click
from ai_craftsman_kb.config import load_config

@click.group()
@click.option('--config-dir', type=click.Path(), default=None,
              help='Path to config directory (default: ~/.ai-craftsman-kb/)')
@click.pass_context
def cli(ctx: click.Context, config_dir: str | None) -> None:
    """AI Craftsman KB — local content aggregation and search."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(Path(config_dir) if config_dir else None)

# ── Ingest ──────────────────────────────────────────────────────────────────

@cli.command('ingest')
@click.option('--source', type=str, default=None,
              help='Ingest only this source type (e.g. hn, substack)')
@click.pass_context
def ingest_pro(ctx, source: str | None) -> None:
    """Pull latest content from all enabled pro-tier sources."""

# ── Search ───────────────────────────────────────────────────────────────────

@cli.command('search')
@click.argument('query')
@click.option('--source', type=str, multiple=True,
              help='Filter by source type (repeatable)')
@click.option('--since', type=str, default=None,
              help='Only results after this date (e.g. 2025-01-01)')
@click.option('--limit', type=int, default=20)
@click.option('--mode', type=click.Choice(['hybrid', 'semantic', 'keyword']),
              default='hybrid')
@click.pass_context
def search(ctx, query: str, source, since, limit, mode) -> None:
    """Search indexed content."""

# ── Ingest URL ───────────────────────────────────────────────────────────────

@cli.command('ingest-url')
@click.argument('url')
@click.option('--tag', type=str, multiple=True, help='Tag to apply (repeatable)')
@click.pass_context
def ingest_url(ctx, url: str, tag) -> None:
    """Ingest a single URL into the index."""

# ── Entities ─────────────────────────────────────────────────────────────────

@cli.command('entities')
@click.option('--type', 'entity_type', type=str, default=None,
              help='Filter by entity type (person, company, technology, ...)')
@click.option('--top', type=int, default=20)
@click.pass_context
def entities(ctx, entity_type, top) -> None:
    """List top entities by mention count."""

# ── Radar ─────────────────────────────────────────────────────────────────────

@cli.command('radar')
@click.argument('query')
@click.option('--source', type=str, multiple=True,
              help='Limit to these source types (repeatable)')
@click.option('--since', type=str, default=None)
@click.pass_context
def radar(ctx, query: str, source, since) -> None:
    """Search the open web on-demand for a topic."""

# ── Briefing ─────────────────────────────────────────────────────────────────

@cli.command('briefing')
@click.argument('topic')
@click.option('--run-radar/--no-radar', default=True)
@click.option('--run-ingest/--no-ingest', default=True)
@click.pass_context
def briefing(ctx, topic: str, run_radar, run_ingest) -> None:
    """Generate a content briefing on a topic."""

# ── Server ───────────────────────────────────────────────────────────────────

@cli.command('server')
@click.option('--host', default='127.0.0.1')
@click.option('--port', type=int, default=None,
              help='Backend port (default from settings.yaml)')
@click.pass_context
def server(ctx, host, port) -> None:
    """Start the FastAPI backend + dashboard."""

# ── Stats ─────────────────────────────────────────────────────────────────────

@cli.command('stats')
@click.pass_context
def stats(ctx) -> None:
    """Show system statistics."""

# ── Doctor ───────────────────────────────────────────────────────────────────

@cli.command('doctor')
@click.pass_context
def doctor(ctx) -> None:
    """Check system health: API keys, DB, Qdrant, config."""
```

**`pyproject.toml` addition**:
```toml
[project.scripts]
cr = "ai_craftsman_kb.cli:cli"
```

**Startup initialization** (shared by all commands via context):
```python
def _get_db_path(config: AppConfig) -> Path:
    return Path(config.settings.data_dir).expanduser() / 'data.db'

def _ensure_data_dir(config: AppConfig) -> Path:
    data_dir = Path(config.settings.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
```

**Pattern for commands** — each command:
1. Gets `config = ctx.obj['config']`
2. Calls `asyncio.run(async_impl(config, ...))`
3. All business logic lives in async functions in other modules; CLI is thin orchestration

## Dependencies
- Depends on: task_02 (load_config, AppConfig), task_03 (init_db for db-touching commands)
- Packages needed: `click` (already in pyproject.toml)

## Acceptance Criteria
- [ ] `cr --help` shows all commands
- [ ] `cr search "hello"` runs without error (can print "not implemented" for now)
- [ ] `cr --config-dir /tmp/test ingest` loads config from the specified dir
- [ ] `cr stats` connects to DB and prints row counts
- [ ] All commands accessible via `uv run cr <command>`
- [ ] `pyproject.toml` includes the `cr` entry point script
- [ ] CLI tests use `click.testing.CliRunner` to invoke commands without subprocess

## Notes
- In task_08 the command bodies can be stubs (`click.echo('TODO')`) — full implementation in later tasks
- `asyncio.run()` is the bridge between Click's sync context and async business logic
- The `--config-dir` option at group level makes config available to all subcommands via `ctx.obj`
- Later tasks (09, 17, 25, etc.) fill in the actual implementations
- Use `click.echo()` for output throughout; `rich` upgrades come in task_43
