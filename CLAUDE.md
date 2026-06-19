# miru

Multimodal reasoning tracer — extract, visualize, and explain what vision-language models attend to. Attention maps, reasoning traces, visualization overlays, and a dataset recorder for training data collection.

**v1.9.0** — 733 tests passing (5 skipped without `MIRU_TEST_REAL_BACKENDS=1`).

## Stack
Python 3.10+ · FastAPI · Pydantic v2 · transformers (CLIP, optional) · Pillow · NumPy · uvicorn

## Commands
```bash
python -m pytest tests/ -x                           # full test suite (mock backends)
MIRU_TEST_REAL_BACKENDS=1 python -m pytest tests/    # include real VLM backend tests
uvicorn miru.main:app --reload                        # dev server on :8000
python -m miru                                        # CLI entry point
```

## Critical Constraints
- No `unwrap()` — raise with a clear message or log + re-raise
- No silent failures — `logging.warning` when a fallback path swallows an error
- `transformers` / `torch` are **optional** — all code paths must work in mock-only mode
- `CLIPBackend` must lazy-load weights on first `infer()` call — never at import time
- Visualization fallback: pure-zlib PNG encoder when Pillow is absent — never fail silently
- Real-backend tests are always gated behind `MIRU_TEST_REAL_BACKENDS=1` — CI runs offline
- Attention maps must always be float32 with values in [0, 1] — assert at `AttentionExtractor` output
- Version bumps touch `pyproject.toml` + `miru/__init__.py`

## Module Map
| Module | Role |
|--------|------|
| `miru/api/routes.py` | FastAPI app: `GET /health`, `POST /analyze` |
| `miru/models/base.py` | `VLMBackend` abstract interface |
| `miru/models/mock.py` | Deterministic `MockVLMBackend` (stable-hash Gaussian attention) |
| `miru/models/clip.py` | `CLIPBackend` — HuggingFace CLIP via transformers (optional) |
| `miru/models/registry.py` | `register()`, `get()`, `available()`, `register_defaults()` |
| `miru/attention/extractor.py` | Min-max norm, block-average resize, top-k hotspot detection |
| `miru/reasoning/tracer.py` | Structured reasoning trace with decay confidence |
| `miru/visualization/overlay.py` | Attention heatmap overlay on input image |
| `miru/recorder.py` | Dataset recorder for training data collection |
| `miru/fidelity.py` | Deletion-test fidelity scorecard (does masking salient pixels drop confidence?) |
| `miru/synergy.py` | Modality-level vision×language Shapley-interaction probe (F_syn) |
| `miru/alerts.py` | SQLite rule store + webhook delivery for `/explain` anomalies |
| `api/main.py` | Deployable FastAPI surface: `/explain`, `/explain/batch`, `/annotate`, etc. |
| `miru/cli/` | CLI entry points |

## Planning Docs
- `PLAN.md` — current phase state and version history
- `CHANGELOG.md` — all notable changes

## Konjo Quality Framework

Three walls against AI slop — all enforced by CI.

**Wall 1 — Pre-commit** (`bash .konjo/scripts/install-hooks.sh` to activate):
ruff lint, ruff format, bare-except scan, DRY check, TODO scan. Blocks the commit.

**Wall 2 — CI gate** (`.github/workflows/konjo-gate.yml`):
Coverage ≥ 80% · mutation survival ≤ 10% · complexity ≤ 15 · file ≤ 500L · zero DRY violations. Blocks the merge.

**Wall 3 — Adversarial review** (local only — disabled in CI):
`git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py`
Claude Opus adversarial critic answers 10 mandatory quality questions.

See `KONJO_QUALITY_FRAMEWORK.md` for the full specification.

## Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).
