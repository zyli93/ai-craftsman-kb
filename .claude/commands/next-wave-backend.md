---
description: "Dispatch backend Python tasks"
---

# Next Wave — Backend

## Scope
You ONLY handle tasks that involve Python backend code:
- Ingestors (tasks 06, 10-16)
- Database, config, LLM providers (tasks 02-05, 07)
- Search, embeddings, entities (tasks 18-25)
- Radar engine (tasks 26-29)
- FastAPI + MCP server (tasks 30-31)
- Briefing, discovery, polish (tasks 41-45)

You do NOT touch: dashboard scaffolding, React pages, or frontend components.

## Steps
1. Read `.claude/tasks/STATUS.md`
2. Find all 🔲 todo tasks within your scope
3. Check dependencies — only show tasks where ALL deps are ✅ or 🔀
4. Show me the ready tasks and ASK which to run
5. After I confirm, IMMEDIATELY mark chosen tasks as 🔵 in-progress in STATUS.md
   and commit that change so the other runner sees it
6. Dispatch subagents with worktree isolation
7. Each subagent must:
   - Read its task file
   - `git checkout main && git pull origin main` first
   - Create branch and implement
   - Rebase onto main before finishing
   - Commit with conventional messages
8. Report results, update STATUS.md

Do NOT merge branches. The monitor session handles that.
