# Task 29: Radar CLI + Promote/Archive/Delete

## Wave
Wave 12 (sequential — depends on tasks 08 and 26)
Domain: backend

## Objective
Implement the `cr radar` CLI command and the triage actions (promote, archive, delete) for radar results, replacing the task_08 stub with a working implementation.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/cli.py` — Implement `radar` command body
- `backend/tests/test_cli.py` — Add radar CLI tests

### Key interfaces / implementation details:

**`radar` command** (replace task_08 stub):
```python
@cli.command('radar')
@click.argument('query')
@click.option('--source', type=str, multiple=True,
              help='Limit to these source types (e.g. hn reddit arxiv devto youtube)')
@click.option('--since', type=str, default=None,
              help='Only results after this date (ISO format)')
@click.option('--limit', type=int, default=10,
              help='Max results per source (default 10)')
@click.pass_context
def radar(ctx, query, source, since, limit): ...
```

**Additional triage sub-commands** (new commands in cli.py):
```python
@cli.command('promote')
@click.argument('document_id')
@click.pass_context
def promote(ctx, document_id: str) -> None:
    """Promote a radar result to pro tier (set promoted_at timestamp)."""

@cli.command('archive')
@click.argument('document_id')
@click.pass_context
def archive(ctx, document_id: str) -> None:
    """Archive a document (hide from default views)."""

@cli.command('delete')
@click.argument('document_id')
@click.confirmation_option(prompt='Delete this document permanently?')
@click.pass_context
def delete(ctx, document_id: str) -> None:
    """Soft-delete a document (set deleted_at timestamp)."""
```

**Radar search output format**:
```
Searching across: hn, reddit, arxiv, devto, youtube...

  1. [ARXIV] "Group Relative Policy Optimization for LLMs"
     arxiv.org/abs/2501.12345 · Jan 15, 2025
     Abstract: "We present GRPO, a novel approach to..."
     Document ID: abc-123

  2. [HN  ] "Show HN: GRPO implementation in PyTorch"
     news.ycombinator.com/item?id=42000000 · Jan 16, 2025 · 142 points
     Document ID: def-456

  ...

Found 18 new, 3 duplicates. Run `cr promote <id>` to add to pro tier.
```

**`_run_radar()` implementation**:
```python
async def _run_radar(config, query, source, since, limit) -> None:
    ingestors = _build_ingestors(config)   # instantiate all ingestors
    async with get_db(_get_db_path(config)) as conn:
        engine = RadarEngine(config, ingestors)
        report = await engine.search(
            conn=conn,
            query=query,
            sources=list(source) if source else None,
            limit_per_source=limit,
        )
    _print_radar_report(report, conn)
```

**Triage actions** (DB operations):
```python
async def _run_promote(config, document_id: str) -> None:
    """Set documents.promoted_at = now() for the given document_id."""

async def _run_archive(config, document_id: str) -> None:
    """Set documents.is_archived = True for the given document_id."""

async def _run_soft_delete(config, document_id: str) -> None:
    """Set documents.deleted_at = now() for the given document_id."""
```

Add corresponding DB query functions to `queries.py`:
```python
async def promote_document(conn, doc_id: str) -> None: ...
async def archive_document(conn, doc_id: str) -> None: ...
async def soft_delete_document(conn, doc_id: str) -> None: ...
```

## Dependencies
- Depends on: task_08 (CLI stubs), task_26 (RadarEngine)
- Packages needed: none new

## Acceptance Criteria
- [ ] `cr radar "GRPO paper"` fans out to all sources and prints results
- [ ] `cr radar "AI" --source hn arxiv` limits to HN and ArXiv
- [ ] `cr radar "AI" --limit 5` returns max 5 results per source
- [ ] `cr promote <id>` sets `promoted_at` on the document
- [ ] `cr archive <id>` sets `is_archived = True`
- [ ] `cr delete <id>` prompts for confirmation, then sets `deleted_at`
- [ ] Output shows document_id that can be copy-pasted for triage commands
- [ ] CLI tests use CliRunner with mocked RadarEngine

## Notes
- The `cr radar` command intentionally runs radar search AND stores results in one call
- After promoting a radar result, it appears in regular search results (pro + radar mix)
- Soft-delete sets `deleted_at` — documents are excluded from search but not erased from DB
- `click.confirmation_option` adds `--yes` flag to skip prompt in scripts
