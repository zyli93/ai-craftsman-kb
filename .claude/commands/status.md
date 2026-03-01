---
description: "Full status dashboard"
---

# Status Check

1. `git pull origin main` first
2. Read `.claude/tasks/STATUS.md`
3. Show:
```
═══ AI Craftsman KB — Build Status ═══

Progress: 12/45 done (27%)
██████░░░░░░░░░░░░░░ 

Backend:  8/33 done    Frontend: 4/12 done

🔵 In-progress:
  task_13 - Reddit ingestor (backend runner)
  task_14 - ArXiv ingestor (backend runner)
  task_32 - Dashboard scaffold (frontend runner)

✅ Ready to merge:
  task_10 - Substack ingestor → branch: task/10-substack
  task_11 - RSS ingestor → branch: task/11-rss

🔲 Ready to start (deps met):
  task_15 - DEV.to ingestor
  task_16 - Adhoc URL ingestor

⏳ Waiting (deps not met):
  task_17 - Phase 2 integration (needs: 10-16)
  task_33 - Overview page (needs: 30, 32)

❌ Blocked: (none)

Next bottleneck: task_30 (REST API) blocks all dashboard pages
```

4. List branches ready for merge: `git branch --list "task/*"`
5. Show any branches that need rebase (behind main)
