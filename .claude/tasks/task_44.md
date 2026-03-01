# Task 44: Export Functionality

## Wave
Wave 16 (parallel with tasks 42, 43, 45; depends on task 21)
Domain: backend

## Objective
Implement export functionality: search results and document lists exportable as Markdown or JSON from both the CLI and API, and briefings exportable as Markdown files.

## Scope

### Files to create/modify:
- `backend/ai_craftsman_kb/export.py` — Export formatters
- `backend/ai_craftsman_kb/cli.py` — Add `--output` and `--format` flags to `search` command
- `backend/ai_craftsman_kb/api/search.py` — Add export content-type support to `GET /api/search`

### Key interfaces / implementation details:

**Export formats** (from plan.md):
- Search results → Markdown (with metadata, entities, citations)
- Briefings → Markdown or JSON
- Documents → Markdown or JSON (bulk)
- Entities → CSV or JSON

**`export.py`**:
```python
class ExportFormat(str, Enum):
    markdown = 'markdown'
    json = 'json'
    csv = 'csv'

def search_results_to_markdown(
    results: list[SearchResult],
    query: str,
    generated_at: str,
) -> str:
    """Format search results as Markdown.

    # Search Results: "{query}"
    Generated: {generated_at}

    ---

    ## 1. {title}
    **Source**: {source_type} | **Date**: {published_at} | **Score**: {score:.2f}
    **URL**: {url}
    **Author**: {author}

    {excerpt}

    ---
    """

def search_results_to_json(results: list[SearchResult]) -> str:
    """JSON array of result dicts (same fields as API response)."""

def documents_to_markdown(docs: list[DocumentRow]) -> str:
    """Bulk document export as concatenated Markdown files."""

def documents_to_json(docs: list[DocumentRow]) -> str:
    """JSON array of document dicts."""

def entities_to_csv(entities: list[EntityRow]) -> str:
    """CSV: id, name, entity_type, mention_count, first_seen_at"""

def entities_to_json(entities: list[EntityRow]) -> str:
    """JSON array of entity dicts."""

def briefing_to_markdown(briefing: BriefingRow) -> str:
    """Return briefing.content (already markdown) with a header block."""

def briefing_to_json(briefing: BriefingRow) -> dict:
    """Return briefing as dict with all fields."""
```

**CLI export flags** (add to `search` command):
```python
@cli.command('search')
@click.argument('query')
...
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Write output to file instead of stdout')
@click.option('--format', 'fmt', type=click.Choice(['markdown', 'json']),
              default=None, help='Output format (default: pretty terminal output)')
```
When `--format` specified: call appropriate exporter, write to `--output` file or stdout.

**API export** — accept format via `Accept` header or `?format=` query param:
```python
@router.get('/search')
async def search(..., format: str | None = None):
    results = await hybrid_search.search(...)
    if format == 'markdown':
        content = search_results_to_markdown(results, q, now())
        return Response(content, media_type='text/markdown')
    elif format == 'json':
        # default FastAPI JSON response
    return results  # normal JSON
```

**`cr briefing` export** (add `--output` to briefing command):
```python
@cli.command('briefing')
@click.argument('topic')
...
@click.option('--output', '-o', type=click.Path(), default=None)
```
When `--output` provided: write `briefing.content` to file. Default: print to terminal.

**File naming convention**:
- Search: `search-{sanitized_query}-{date}.md`
- Briefing: `briefing-{sanitized_topic}-{date}.md`
- Documents: `documents-export-{date}.json`
- Entities: `entities-{date}.csv`

## Dependencies
- Depends on: task_21 (SearchResult model), task_25 (search CLI)
- Packages needed: `csv` stdlib (already available)

## Acceptance Criteria
- [ ] `cr search "LLM" --format markdown --output results.md` writes Markdown file
- [ ] `cr search "LLM" --format json` prints JSON to stdout
- [ ] `cr briefing "AI agents" --output brief.md` writes briefing to file
- [ ] Markdown search output includes: title, score, URL, excerpt, metadata
- [ ] JSON export is valid JSON parseable by `json.loads()`
- [ ] Entities CSV has header row: `id,name,entity_type,mention_count,first_seen_at`
- [ ] `GET /api/search?q=test&format=markdown` returns Markdown with `text/markdown` content-type
- [ ] File paths sanitized (spaces → underscores, special chars removed)

## Notes
- Export is purely formatting — no new data fetching needed
- Markdown output should work as a standalone document (include query + date in header)
- For briefing export, `briefing.content` is already Markdown — just add a YAML frontmatter header
- The `csv` module is stdlib — no extra dependencies needed
- Bulk document export (`cr search --format json --limit 1000`) can be used for backup/portability
