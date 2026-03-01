# Task 25: Search CLI Enhancements

## Wave
Wave 9 (sequential — depends on tasks 08, 21, 23)
Domain: backend

## Objective
Implement the full `cr search` and `cr entities` CLI commands, replacing the task_08 stubs with working implementations that call `HybridSearch` and `EntitySearch` and display results clearly.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/cli.py` — Implement `search` and `entities` command bodies
- `backend/tests/test_cli.py` — Add tests for search and entity commands

### Key interfaces / implementation details:

**`search` command** implementation:
```python
async def _run_search(
    config: AppConfig,
    query: str,
    source: tuple[str, ...],
    since: str | None,
    limit: int,
    mode: str,
) -> None:
    async with get_db(_get_db_path(config)) as conn:
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
        click.echo('No results found.')
        return

    for i, r in enumerate(results, 1):
        click.echo(f'\n{i}. {r.title or "(no title)"}')
        click.echo(f'   {r.source_type} · {r.author or "unknown"} · {r.published_at or "no date"}')
        click.echo(f'   {r.url}')
        if r.excerpt:
            click.echo(f'   "{r.excerpt[:200]}..."')
        click.echo(f'   Score: {r.score:.3f} | Origin: {r.origin}')
```

**`entities` command** implementation:
```python
async def _run_entities(
    config: AppConfig,
    entity_type: str | None,
    top: int,
) -> None:
    async with get_db(_get_db_path(config)) as conn:
        entity_search = EntitySearch()
        entities = await entity_search.list_entities(
            conn, entity_type=entity_type, limit=top
        )

    if not entities:
        click.echo('No entities found.')
        return

    click.echo(f'\nTop {len(entities)} entities'
               f'{f" (type: {entity_type})" if entity_type else ""}:\n')
    for e in entities:
        click.echo(f'  [{e.entity_type:12s}] {e.name:30s}  {e.mention_count} mentions')
```

**`stats` command** implementation (also implement here):
```python
async def _run_stats(config: AppConfig) -> None:
    async with get_db(_get_db_path(config)) as conn:
        stats = await get_stats(conn)
    vector_store = VectorStore(config)
    qdrant_info = vector_store.get_collection_info()

    click.echo('\nAI Craftsman KB — System Stats')
    click.echo(f'  Documents     : {stats["total_documents"]}')
    click.echo(f'  Embedded      : {stats["embedded_documents"]} / {stats["total_documents"]}')
    click.echo(f'  Entities      : {stats["total_entities"]}')
    click.echo(f'  Sources       : {stats["total_sources"]}')
    click.echo(f'  Briefings     : {stats["total_briefings"]}')
    click.echo(f'  Vectors       : {qdrant_info.get("vectors_count", 0)}')
```

**Output format**: Plain `click.echo()` for now. Task_43 upgrades to `rich` tables. Keep output concise — avoid word wrap issues in terminal.

**`--mode` validation**: Click choice validated at CLI level. If `mode='semantic'` but Qdrant is empty, print a helpful message: "No embeddings found. Run `cr ingest` first."

## Dependencies
- Depends on: task_08 (CLI stubs), task_21 (HybridSearch), task_23 (EntitySearch)
- Packages needed: none new

## Acceptance Criteria
- [ ] `cr search "LLM inference"` returns results in < 2s for 1000-document DB
- [ ] `cr search "LLM" --source hn --mode keyword` filters correctly
- [ ] `cr search "AI" --since 2025-01-01` filters by date
- [ ] `cr entities` lists top 20 entities by mention count
- [ ] `cr entities --type person` lists only person entities
- [ ] `cr stats` shows correct counts matching DB contents
- [ ] CLI tests use `CliRunner` and mock DB/VectorStore to avoid real I/O
- [ ] Graceful message when no results or empty DB

## Notes
- `since` parameter passed as string (e.g. "2025-01-01") — validate as ISO date in CLI before passing to search
- `click.echo()` output should be readable in a terminal without rich formatting — task_43 adds pretty output
- The `stats` command implementation moved here (from task_08 stub) since it now needs Qdrant info
