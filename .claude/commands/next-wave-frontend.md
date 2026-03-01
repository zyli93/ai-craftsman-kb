---
description: "Dispatch frontend dashboard tasks"
---

# Next Wave — Frontend

## Scope
You ONLY handle tasks that involve the React dashboard:
- Dashboard scaffolding (task 32)
- All dashboard pages (tasks 33-39)
- Server startup command (task 40)

You do NOT touch: Python backend, ingestors, search, CLI, or API code.

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
   - Run `pnpm build` to verify no errors
   - Rebase onto main, resolve any conflicts
   - Re-run `pnpm build` after rebase
   - Fast-forward merge into main, push, delete branch
   - Update STATUS.md to 🔀 merged and push
   - Commit with conventional messages
8. Report results — branches should already be merged when agents complete
