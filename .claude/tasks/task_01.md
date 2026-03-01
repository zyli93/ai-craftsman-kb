# Task 01: Project Scaffolding

## Objective
Set up the Python project with uv, directory structure, and base dependencies.

## Scope
### Files to create:
- `pyproject.toml` — Python project config with uv
- `backend/ai_craftsman_kb/__init__.py`
- `backend/ai_craftsman_kb/cli.py` — Empty Click app placeholder
- `backend/ai_craftsman_kb/config/__init__.py`
- `backend/ai_craftsman_kb/db/__init__.py`
- `backend/ai_craftsman_kb/llm/__init__.py`
- `backend/ai_craftsman_kb/ingestors/__init__.py`
- `backend/ai_craftsman_kb/processing/__init__.py`
- `backend/ai_craftsman_kb/search/__init__.py`
- `backend/ai_craftsman_kb/radar/__init__.py`
- `backend/ai_craftsman_kb/briefing/__init__.py`
- `backend/tests/__init__.py`
- `dashboard/.gitkeep`
- `config/sources.yaml` — Default source config
- `config/settings.yaml` — Default settings
- `config/filters.yaml` — Default filter config
- `.gitignore`
- `README.md`

### Dependencies to include in pyproject.toml:

click, httpx, pydantic, pyyaml, readability-lxml, lxml, html2text,
openai, tiktoken, aiosqlite, qdrant-client, feedparser, rich

## Acceptance Criteria
- [ ] `uv sync` succeeds
- [ ] `uv run python -c "import ai_craftsman_kb"` works
- [ ] Directory structure matches plan.md
- [ ] .gitignore covers Python, Node, .env, ~/.ai-craftsman-kb/

## Notes
- Use Python 3.12+ as minimum version
- Use `src` layout is NOT needed — flat `backend/ai_craftsman_kb/` is fine
- Read plan.md for the full directory structure
