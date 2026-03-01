---
description: "Show ready tasks and dispatch a wave of parallel agents"
---

# Next Wave

## Step 1: Analyze
- Read `.claude/tasks/STATUS.md`
- For each 🔲 todo task, check if all dependencies are ✅ done or 🔀 merged
- List all READY tasks with their descriptions
- Check for file overlap between ready tasks

## Step 2: Present the Wave
Show me a table like this:
```
Ready tasks (no dependency blocking):
  task_10 - Substack ingestor
  task_11 - RSS ingestor  
  task_12 - YouTube ingestor
  task_13 - Reddit ingestor

⚠️  Deferred (file overlap with above):
  (none)

Still waiting:
  task_17 - Phase 2 integration (needs 10-16)
```

Then ASK ME which tasks to run. Wait for my confirmation before dispatching.

## Step 3: Dispatch
After I confirm, for each approved task:
- Spawn a subagent with worktree isolation
- Each subagent reads its task file, creates a branch, implements, tests, commits
- Each subagent must run the git workflow from CLAUDE.md before starting

## Step 4: Report
After all subagents complete, show results:
- ✅ Succeeded: list with branch names
- ❌ Failed: list with reasons
- Update STATUS.md accordingly

Do NOT merge. Do NOT push. Just report. I'll handle merging from my other session.
