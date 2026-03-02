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

### After completing work — agents self-merge:
Agents must rebase, test, merge, push, and clean up their own branch.
Do NOT leave branches open waiting for another session to merge.

```bash
cd /Users/zeyuli/projects/ai-craftsman-kb

# 1. Rebase onto latest main
git fetch origin
git checkout main && git pull origin main
git checkout task/XX-short-description
git rebase origin/main
```

If rebase conflicts:
1. Read both sides to understand what changed
2. Keep the more complete/correct version, preserving changes from both sides
3. `git add <resolved-file>` then `git rebase --continue`
4. If too complex to resolve confidently: `git rebase --abort`, mark task ❌ blocked in STATUS.md, report — do NOT force through

```bash
# 2. Run full tests after rebase
uv run pytest backend/tests/ -v        # backend
# cd dashboard && pnpm build           # frontend

# 3. Fast-forward merge + push
git checkout main
git merge task/XX-short-description --ff-only
git push origin main

# 4. Clean up branch
git branch -d task/XX-short-description

# 5. Mark merged in STATUS.md, commit + push
git add .claude/tasks/STATUS.md
git commit -m "status: mark task_XX as merged"
git push origin main
```

### NEVER:
- Force push to main
- Merge without rebasing and testing first
- Leave unresolved conflicts
- Leave a branch open after a successful merge

## Current Status
Check .claude/tasks/STATUS.md for task status tracker.

## Task Files
Each task is defined in `.claude/tasks/task_XX.md`. Read the task file
before starting work. Update STATUS.md when you begin and complete a task.

## Context Management

**One task per context window.** Clear context after each task completes.

- All project state lives in files (STATUS.md, CLAUDE.md, task files, code)
- Never rely on remembering code from earlier in the conversation — always re-read
- Stale context causes drift: hallucinated file contents, wrong assumptions, inconsistencies
- Worktree-isolated subagents already get fresh context per task — persistent sessions should behave the same way

Pattern:
```
start task → read task file → read relevant source files → implement → push → CLEAR
```

## IMPORTANT
- Read plan.md for full architecture details before starting any task.
- Read the specific task file completely before writing any code.
- Make atomic git commits with conventional commit messages.
- Do NOT modify files outside your task's scope.
- Update STATUS.md when starting and completing tasks.
