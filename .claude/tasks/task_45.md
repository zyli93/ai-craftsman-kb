# Task 45: Error Resilience + `cr doctor`

## Wave
Wave 16 (parallel with tasks 42, 43, 44; depends on all prior tasks)
Domain: backend

## Objective
Add structured error handling and retry logic across the ingest pipeline, and implement the `cr doctor` command that validates system health (API keys, DB, Qdrant, config, connectivity) and reports issues with actionable fixes.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/cli.py` — Implement `doctor` command body
- `backend/ai_craftsman_kb/resilience.py` — Retry decorator + error categorization
- `backend/ai_craftsman_kb/api/system.py` — Enhance `GET /api/health` with full diagnostics

### Key interfaces / implementation details:

**`resilience.py`** — shared retry + error handling:
```python
import asyncio
from functools import wraps

class AppError(Exception):
    """Base class for application errors."""
    def __init__(self, message: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.recoverable = recoverable

class APIError(AppError):
    """External API call failed."""
    def __init__(self, provider: str, status_code: int, message: str) -> None: ...

class ConfigError(AppError):
    """Configuration is invalid or missing."""
    def __init__(self, field: str, message: str) -> None:
        super().__init__(message, recoverable=False)

class QuotaExceededError(APIError):
    """API quota exceeded — cannot retry."""
    pass

def retry_async(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    retry_on: tuple[type[Exception], ...] = (APIError,),
    skip_on: tuple[type[Exception], ...] = (QuotaExceededError, ConfigError),
) -> callable:
    """Decorator: retry async function with exponential backoff.
    Retries only on retryable exceptions; immediately raises non-retryable ones.

    Backoff: base * 2^(attempt-1) seconds, capped at backoff_max.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except skip_on:
                    raise
                except retry_on as e:
                    if attempt == max_attempts:
                        raise
                    delay = min(backoff_base * 2 ** (attempt - 1), backoff_max)
                    await asyncio.sleep(delay)
        return wrapper
    return decorator
```

**Apply `@retry_async`** to:
- All LLM provider `complete()` and `embed()` methods (task_04)
- All ingestor HTTP fetch calls (after this task, or as a note for code review)
- `ContentExtractor.fetch_and_extract()` (task_05)

**`cr doctor` command**:
```python
@cli.command('doctor')
@click.pass_context
def doctor(ctx) -> None:
    """Check system health and configuration."""
    asyncio.run(_run_doctor(ctx.obj['config']))
```

**`_run_doctor()` checks** (from plan.md diagnostics):
```python
async def _run_doctor(config: AppConfig) -> None:
    checks = [
        ('Config file', _check_config(config)),
        ('SQLite DB', _check_database(config)),
        ('Qdrant', _check_qdrant(config)),
        ('OpenAI API key', _check_api_key(config, 'openai')),
        ('Anthropic API key', _check_api_key(config, 'anthropic')),
        ('OpenRouter API key', _check_api_key(config, 'openrouter')),
        ('YouTube API key', _check_youtube_key(config)),
        ('Reddit credentials', _check_reddit_credentials(config)),
        ('HN connectivity', _check_hn_connectivity()),
        ('ArXiv connectivity', _check_arxiv_connectivity()),
        ('Data directory', _check_data_dir(config)),
    ]

    all_ok = True
    for name, coro in checks:
        status, message = await coro
        icon = '✓' if status == 'ok' else ('⚠' if status == 'warn' else '✗')
        color = 'green' if status == 'ok' else ('yellow' if status == 'warn' else 'red')
        console.print(f'  [{color}]{icon}[/{color}] {name:30s} {message}')
        if status == 'error':
            all_ok = False

    if all_ok:
        console.print('\n[green]All checks passed. System ready.[/green]')
    else:
        console.print('\n[red]Some checks failed. See messages above.[/red]')
        raise SystemExit(1)
```

**Individual checks**:
```python
async def _check_database(config) -> tuple[str, str]:
    """Open DB connection, run 'SELECT COUNT(*) FROM documents'. Return doc count."""

async def _check_qdrant(config) -> tuple[str, str]:
    """Initialize VectorStore, call get_collection_info(). Return vector count."""

async def _check_api_key(config, provider: str) -> tuple[str, str]:
    """Check that api_key for provider is set (non-empty).
    Return 'warn' if not set (provider may not be needed), not 'error'."""

async def _check_connectivity(url: str, name: str) -> tuple[str, str]:
    """GET url with 5s timeout. Return 'ok' if 200, 'error' if fails."""

async def _check_data_dir(config) -> tuple[str, str]:
    """Check data_dir exists and is writable. Show path + disk free space."""
```

**Enhanced `GET /api/health`**:
```python
@router.get('/health')
async def health(request: Request, full: bool = False):
    result = {'status': 'ok', 'db': True, 'qdrant': True}
    if full:
        # Run subset of doctor checks
        result['checks'] = {...}
    return result
```

## Dependencies
- Depends on: all prior tasks (uses config, DB, Qdrant, LLM providers)
- Packages needed: none new

## Acceptance Criteria
- [ ] `cr doctor` runs all checks and prints colored pass/fail/warn per check
- [ ] Missing API key → `⚠ warn` (not error, since some providers are optional)
- [ ] DB unreachable → `✗ error`
- [ ] Qdrant path doesn't exist → `✗ error` with "Run `cr ingest` to initialize"
- [ ] All checks complete even if some fail (no early exit on first failure)
- [ ] `cr doctor` exits with code 1 if any check errors, 0 if all pass or warn
- [ ] `@retry_async` decorator correctly retries failed LLM calls with backoff
- [ ] Retry skipped for `QuotaExceededError` and `ConfigError`
- [ ] `GET /api/health?full=true` returns extended diagnostics

## Notes
- `cr doctor` is the first command new users should run after setup — make the output helpful and actionable. Each error message should include a suggested fix (e.g. "Set OPENAI_API_KEY in your environment").
- The retry decorator is applied at the provider level (task_04), not the ingestor level — ingestors may have source-specific retry logic
- Doctor checks are `(status, message)` tuples: `status` is `'ok'` | `'warn'` | `'error'`
- Network checks use a 5-second timeout — doctor should complete in < 15 seconds total
- The `SystemExit(1)` at the end allows CI scripts to detect failures: `cr doctor || exit 1`
