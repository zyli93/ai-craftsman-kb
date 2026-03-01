---
description: "Start working on a specific task from the task tracker"
argument-hint: "<task_number, e.g. 01>"
---

# Work on Task $ARGUMENTS

## Steps:

1. `git pull origin main` to get latest
2. Read `.claude/tasks/STATUS.md` to check current status
3. Read `.claude/tasks/task_$ARGUMENTS.md` for full task details
4. Read `plan.md` for architecture context
5. Check that all dependencies listed in the task are ✅ done or 🔀 merged in STATUS.md
6. If dependencies are not done, STOP and report which are missing
7. Update STATUS.md: change task_$ARGUMENTS status to 🔵 in-progress
8. Commit and push the status change:
```
   git add .claude/tasks/STATUS.md
   git commit -m "status: claim task_$ARGUMENTS as in-progress"
   git push origin main
```
9. Create branch: `git checkout -b task/$ARGUMENTS-<short-description>`
10. Implement the task according to its acceptance criteria
11. Run relevant tests
12. Make atomic commits with conventional commit messages
13. Before finishing, rebase onto latest main:
```
    git checkout main && git pull origin main
    git checkout task/$ARGUMENTS-*
    git rebase main
```
14. If rebase conflicts, resolve them preserving both sides
15. Run tests again after rebase
16. Update STATUS.md: change task_$ARGUMENTS status to ✅ done
17. Report what was done and any issues encountered
