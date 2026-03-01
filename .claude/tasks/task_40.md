# Task 40: `cr server` Command (FastAPI + Dashboard)

## Wave
Wave 15 (parallel with tasks 31, 41; depends on tasks 30, 32)
Domain: backend

## Objective
Implement the `cr server` CLI command that starts the FastAPI backend and (optionally) serves the built dashboard static assets, so users can launch the full stack with a single command.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/cli.py` — Implement `server` command body
- `backend/ai_craftsman_kb/server.py` — Add static file serving + MCP startup option

### Key interfaces / implementation details:

**`server` command** (`cli.py`):
```python
@cli.command('server')
@click.option('--host', default='127.0.0.1', show_default=True)
@click.option('--port', type=int, default=None,
              help='Backend port (default from settings.yaml: 8000)')
@click.option('--no-dashboard', is_flag=True, default=False,
              help='Do not serve dashboard static files')
@click.option('--with-mcp', is_flag=True, default=False,
              help='Also start MCP server in same process')
@click.option('--reload', is_flag=True, default=False,
              help='Auto-reload on code changes (development mode)')
@click.pass_context
def server(ctx, host, port, no_dashboard, with_mcp, reload) -> None:
    """Start the FastAPI backend (+ dashboard). Access at http://HOST:PORT"""
    config: AppConfig = ctx.obj['config']
    backend_port = port or config.settings.server.backend_port
    click.echo(f'Starting AI Craftsman KB server at http://{host}:{backend_port}')
    click.echo(f'Dashboard: http://{host}:{backend_port}/')
    click.echo(f'API docs:  http://{host}:{backend_port}/docs')
    import uvicorn
    uvicorn.run(
        'ai_craftsman_kb.server:app',
        host=host,
        port=backend_port,
        reload=reload,
    )
```

**Dashboard static file serving** (`server.py`):
```python
from fastapi.staticfiles import StaticFiles
from pathlib import Path

DASHBOARD_DIST = Path(__file__).parent.parent.parent / 'dashboard' / 'dist'

def mount_dashboard(app: FastAPI) -> None:
    """Mount built dashboard static files at root path.
    Falls back gracefully if dist/ doesn't exist yet."""
    if DASHBOARD_DIST.exists():
        app.mount('/', StaticFiles(directory=DASHBOARD_DIST, html=True), name='dashboard')
    else:
        @app.get('/')
        async def dashboard_not_built():
            return {'message': 'Dashboard not built. Run: cd dashboard && pnpm build'}
```

**`cr mcp` command** (also implement here — simple stdio MCP server):
```python
@cli.command('mcp')
@click.pass_context
def mcp_server(ctx) -> None:
    """Start the MCP server (stdio transport for Claude Desktop)."""
    config: AppConfig = ctx.obj['config']
    from ai_craftsman_kb.mcp_server import run_mcp_server
    run_mcp_server(config)
```

**Startup sequence** in `server.py`:
1. `load_config()` at app startup
2. `init_db(db_path)` — ensures schema exists
3. Initialize shared services: `VectorStore`, `Embedder`, `LLMRouter`
4. Mount dashboard if `dist/` exists
5. Uvicorn starts serving

**Dev workflow** (for `--reload` mode):
- `cr server --reload` — auto-restarts on Python file changes
- Dashboard: run separately with `cd dashboard && pnpm dev` (Vite dev server on port 3000)
- In `--reload` mode, dashboard static files are NOT mounted (dev uses Vite proxy)

## Dependencies
- Depends on: task_30 (FastAPI app), task_32 (dashboard build output)
- Packages needed: `uvicorn[standard]` (add to pyproject.toml)

## Acceptance Criteria
- [ ] `cr server` starts FastAPI on port 8000 (or configured port)
- [ ] `curl http://localhost:8000/api/health` returns `{"status": "ok"}`
- [ ] `cr server --no-dashboard` starts without mounting static files
- [ ] `cr server --reload` enables uvicorn auto-reload
- [ ] `cr mcp` starts MCP server in stdio mode without error
- [ ] If `dashboard/dist/` exists, `/` serves the React app
- [ ] If `dashboard/dist/` missing, `/` returns helpful JSON message
- [ ] `cr server --help` shows all options with descriptions

## Notes
- `uvicorn.run('ai_craftsman_kb.server:app', ...)` takes a string import path — required for `--reload` to work (can't pass a live `app` object with reload)
- Production: `cr server` (no reload) + pre-built dashboard static files
- Development: `cr server --reload` + `pnpm dev` in separate terminal
- MCP stdio transport: MCP client (Claude Desktop) launches `cr mcp` as a subprocess — no port needed
- Add `cr server` and `cr mcp` to Claude Desktop config instructions in README
