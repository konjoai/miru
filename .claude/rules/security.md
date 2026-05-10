---
paths:
  - "**/api*"
  - "**/routes*"
  - "**/middleware*"
---
# Security Rules
- Validate all inputs at the API boundary
- Never log raw user input at INFO level — log a hash or truncated prefix
- Rate-limit all API endpoints by default
- Set per-request timeouts on every operation
- Never store API keys or tokens in the codebase — use environment variables
