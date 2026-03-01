# AI Craftsman KB

## Project Overview
A local-first CLI + web dashboard for aggregating, indexing, and semantically searching
content across HN, Substack, YouTube, Reddit, ArXiv, RSS, and DEV.to.
Two modes: Pro (subscription pulls) and Radar (on-demand topic search).

## Architecture
- **Backend**: Python 3.12+ with FastAPI, SQLite, Qdrant local, Click CLI
- **Dashboard**: TypeScript/React with Vite + Tailwind + shadcn/ui
- **MCP Server**: Python MCP SDK, runs alongside FastAPI
- Package management: `uv` for Python, `pnpm` for dashboard

## Project Structure
See plan.md for full architecture. Key directories:
- `backend/ai_craftsman_kb/` — Python backend
- `dashboard/` — React frontend
- `config/` — Default YAML configs
- `.claude/tasks/` — Task tracking files

## Coding Standards
- Python: Use type hints everywhere. Pydantic models for data. Async where possible.
- Use `httpx` for HTTP, never `requests`.
- SQLite via `aiosqlite`. No ORM.
- React: Functional components only. Tailwind for styling. shadcn/ui components.
- All new code must include docstrings/comments for non-obvious logic.
- Each module should be independently testable.

## Key Commands
- `uv run python -m ai_craftsman_kb.cli ingest pro` — run pro ingestion
- `uv run python -m ai_craftsman_kb.cli search "query"` — search
- `cd dashboard && pnpm dev` — start dashboard dev server

## Git Workflow

### Before starting any task:
git checkout main
git pull origin main
git checkout -b task/XX-short-description

### While working:
Make atomic commits on your feature branch.

### Before reporting completion:
git checkout main
git pull origin main
git checkout task/XX-short-description
git rebase main

If rebase conflicts:
1. Read the conflicting files to understand what changed on main
2. Resolve conflicts preserving BOTH your changes and main's changes
3. git add <resolved files>
4. git rebase --continue
5. If the conflict is too complex, abort with git rebase --abort
   and report the conflict — do NOT force through

After successful rebase:
- Run tests to make sure nothing broke
- Your branch is now cleanly ahead of main, ready for fast-forward merge

### NEVER:
- Force push to main
- Merge without rebasing first
- Leave unresolved conflicts

## Current Status
Check .claude/tasks/STATUS.md for task status tracker.

## Task Files
Each task is defined in `.claude/tasks/task_XX.md`. Read the task file
before starting work. Update STATUS.md when you begin and complete a task.

## IMPORTANT
- Read plan.md for full architecture details before starting any task.
- Read the specific task file completely before writing any code.
- Make atomic git commits with conventional commit messages.
- Do NOT modify files outside your task's scope.
- Update STATUS.md when starting and completing tasks.
