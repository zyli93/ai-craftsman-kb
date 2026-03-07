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
Key directories:
- `backend/ai_craftsman_kb/` — Python backend
- `dashboard/` — React frontend
- `config/` — Default YAML configs
- `doc/` — Setup guide, manual, credentials, vault

## Coding Standards
- Python: Use type hints everywhere. Pydantic models for data. Async where possible.
- Use `httpx` for HTTP, never `requests`.
- SQLite via `aiosqlite`. No ORM.
- React: Functional components only. Tailwind for styling. shadcn/ui components.
- All new code must include docstrings/comments for non-obvious logic.
- Each module should be independently testable.

## Key Commands
- `uv run cr --help` — list all CLI commands
- `uv run cr ingest` — run pro ingestion
- `uv run cr search "query"` — search
- `uv run cr doctor` — health check
- `uv run cr server` — start backend + dashboard
- `cd dashboard && pnpm dev` — start dashboard dev server

See `doc/manual.md` for full CLI reference and operational guide.

## Git Workflow

### Before starting any task:
git checkout main
git pull origin main
git checkout -b task/XX-short-description

### While working:
Make atomic commits on your feature branch.

### After completing work — subagent pushes branch, main agent merges one-by-one:

Two roles, cleanly separated. Subagents never touch main.

**Subagent** (after implementing):
```bash
# 1. Run tests
uv run pytest backend/tests/ -v        # backend
# cd dashboard && pnpm build           # frontend

# 2. Push branch to remote
git push origin task/XX-short-description

# 3. Delete local worktree branch
git checkout main
git branch -d worktree-<agent-id>
```

**Main agent** merges each branch individually — NEVER batch multiple tasks into one merge:
```bash
# 1. Fetch and rebase onto latest main
git fetch origin
git checkout main && git pull origin main
git checkout task/XX-short-description
git rebase origin/main
```

If rebase conflicts:
1. Read both sides to understand what changed
2. Keep the more complete/correct version, preserving changes from both sides
3. `git add <resolved-file>` then `git rebase --continue`
4. If too complex: `git rebase --abort`, report to user

```bash
# 2. Fast-forward merge — preserves individual commits from the branch
git checkout main
git merge task/XX-short-description --ff-only
git push origin main

# 3. Delete remote and local branch
git push origin --delete task/XX-short-description
git branch -d task/XX-short-description
```

### Branch cleanup — stale branches:
Any branch with 0 commits ahead of main is stale — delete immediately:
```bash
git branch --merged main | grep -v "^\* main" | xargs git branch -d
```

### NEVER:
- Force push to main
- Subagents touch or push to main directly
- Batch-merge multiple task branches in one operation
- Squash commits — ff-only preserves the full per-task commit history
- Merge without rebasing and testing first
- Leave unresolved conflicts
- Leave any branch (local or remote) open after work is complete

## Context Management

**One task per context window.** Clear context after each task completes.

- All project state lives in files (CLAUDE.md, doc/, code)
- Never rely on remembering code from earlier in the conversation — always re-read
- Stale context causes drift: hallucinated file contents, wrong assumptions, inconsistencies

## IMPORTANT
- Read `doc/manual.md` for operational details.
- Make atomic git commits with conventional commit messages.
- Do NOT modify files outside your task's scope.
