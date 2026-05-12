---
paths:
  - "**/*.py"
---
# Python Conventions
- No bare `except:` or `except Exception:` — catch specific exceptions; log and re-raise
- No mutable default arguments
- `ruff check` and `ruff format` must be clean before every commit
- `mypy --strict` must be clean
- No dead code — `vulture` zero tolerance
- `radon cc -n C` zero functions above grade C
- `interrogate --fail-under 100` on all public APIs
- Use `logging` not `print` in production code
