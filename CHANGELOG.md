# CHANGELOG

All notable changes to Miru are documented here.  
Format: [Conventional Commits](https://www.conventionalcommits.org/) + [Keep a Changelog](https://keepachangelog.com/).

---

## [0.1.0] — 2026-04-28

### Added

#### Core package (`miru/`)
- `miru/__init__.py` — package entry point; exports `VLMBackend`, `MockVLMBackend`, `AttentionExtractor`, `ReasoningTracer`
- `miru/config.py` — frozen `Settings` Pydantic model with sane defaults (no `pydantic-settings` dependency)
- `miru/schemas.py` — six frozen Pydantic v2 models: `ImageInput`, `AttentionMap`, `ReasoningStep`, `ReasoningTrace`, `HealthResponse`, `ErrorResponse`

#### Models layer (`miru/models/`)
- `miru/models/base.py` — abstract `VLMBackend` ABC and frozen `VLMOutput` dataclass
- `miru/models/mock.py` — deterministic `MockVLMBackend`: stable-hash Gaussian blob attention maps; 5 canned answers; seed-reproducible outputs; no PYTHONHASHSEED dependency (`_stable_hash` uses polynomial rolling hash)

#### Attention layer (`miru/attention/`)
- `miru/attention/extractor.py` — `AttentionExtractor` with `normalize()` (min-max, uniform-safe), `resize_to_grid()` (pure NumPy block averaging), `extract()` (full pipeline), and `top_k_regions()` (argpartition + argsort, O(n + k log k))

#### Reasoning layer (`miru/reasoning/`)
- `miru/reasoning/tracer.py` — `ReasoningTracer.trace()` builds `ReasoningTrace` from `VLMOutput`; per-step confidence decays 5 % per step to model compounding uncertainty

#### API layer (`miru/api/`)
- `miru/api/routes.py` — FastAPI `APIRouter` with `GET /health` and `POST /analyze`; best-effort base64 image decode with 1×1 black pixel fallback; backend registry dict; unknown backend name falls back to default
- `miru/main.py` — `FastAPI` app creation and router registration

#### Tests (`tests/`)
- `tests/conftest.py` — `client` (TestClient) and `mock_image_b64` fixtures
- `tests/test_models.py` — 8 tests: name, output type, confidence range, attention shape, attention normalization, steps non-empty, determinism, different-question variation
- `tests/test_attention.py` — 10 tests: normalize uniform/range/order, resize shape/constant, extract shape/dtype/range, top-k count/order/max/zero-k
- `tests/test_reasoning.py` — 8 tests: type, answer, step count, confidence monotone, attention shape, latency, backend name, step numbering, attention range
- `tests/test_api.py` — 11 tests: health ok/version/backends, analyze success/structure/default-backend/attention-dims/latency/answer/steps, bad-image graceful, unknown-backend fallback

#### Build & CI
- `pyproject.toml` — hatchling build, `[dev]` extras (pytest, pytest-asyncio, httpx, pytest-cov)
- `.github/workflows/ci.yml` — GitHub Actions: Python 3.11, `pip install -e ".[dev]"`, `pytest tests/ -v --tb=short`
- `PLAN.md` — five-phase roadmap (v0.1.0 → v0.5.0)
- `CHANGELOG.md` — this file

### Architecture notes
- Image decode in `/analyze` is intentionally best-effort: invalid payloads silently fall back to a 1×1 black pixel rather than returning a 422, because image validity is not a hard contract at the API boundary.
- `MockVLMBackend._stable_hash` uses a polynomial rolling hash (`h = h*31 + ord(ch)`) to guarantee reproducible outputs independent of `PYTHONHASHSEED`.
- `AttentionExtractor.resize_to_grid` is pure NumPy (no SciPy/PIL) to keep the dependency footprint minimal.
