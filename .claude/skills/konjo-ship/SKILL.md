---
name: konjo-ship
description: Konjo sprint completion checklist for miru.
user-invocable: true
---
# Konjo Ship — miru

## Sprint Completion Checklist
```
[ ] All tests pass — `python -m pytest` green
[ ] `ruff check` and `ruff format --check` clean
[ ] CHANGELOG.md updated
[ ] PLAN.md updated
[ ] README.md reflects current state
[ ] git add && git commit -m "type(scope): description" && git push
```

## Session Handoff Template
```
SHIPPED      [what was completed this session]
TESTS        [passing / failing / count]
PUSHED       [commit hash or "not pushed — reason"]
NEXT SESSION [the exact next task]
DISCOVERIES  [papers, repos, techniques found]
HEALTH       [Green / Yellow / Red]
```
