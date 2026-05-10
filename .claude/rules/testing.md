---
paths:
  - "**/test_*.py"
  - "**/tests/**"
---
# Testing Rules
A sprint is NEVER complete until all tests pass.
Every code file needs a corresponding test file.
Never mock the DB or network in E2E tests.
`python -m pytest` must be green before `git push`.
