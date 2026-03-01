---
description: "Merge a completed task branch into main after rebase"
argument-hint: "<task_number>"
---

# Merge Task $ARGUMENTS

1. Find the branch: `git branch --list "task/$ARGUMENTS-*"`
2. Pull latest main: `git checkout main && git pull origin main`
3. Rebase the task branch onto main:
```
   git checkout task/$ARGUMENTS-*
   git rebase main
```
4. If rebase conflicts:
   - Show me the conflicts and STOP
   - Do NOT auto-resolve — let me decide
5. If rebase succeeds:
   - Run tests: `uv run pytest backend/tests/ -v`
   - If tests pass: fast-forward merge into main
```
     git checkout main
     git merge task/$ARGUMENTS-* --ff-only
```
   - Push: `git push origin main`
6. Clean up branch: `git branch -d task/$ARGUMENTS-*`
7. Update STATUS.md: change task to 🔀 merged
