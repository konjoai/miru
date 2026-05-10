---
paths:
  - "**/benchmarks/**"
  - "**/bench_*.py"
---
# Benchmarking Rules
- Minimum 5 warmup runs before timing.
- Report p50, p95, p99, stddev — not just mean.
- Statistical significance: paired t-test or Wilcoxon signed-rank.
- Results → `benchmarks/results/<timestamp>_<name>/`. Never overwrite.
- Regression gate: >5% p95 latency = hard stop.
