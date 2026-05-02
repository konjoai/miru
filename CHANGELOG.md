# CHANGELOG

All notable changes to Miru are documented here.  
Format: [Conventional Commits](https://www.conventionalcommits.org/) + [Keep a Changelog](https://keepachangelog.com/).

---

## [0.5.0] — 2026-05-02

### Added

#### Dataset recorder (`miru/recorder.py`)
- `TraceRecorder` — threaded JSONL writer with `queue.Queue` + daemon worker; configurable `batch_size` (default 64) and `flush_interval` (default 5s); `start()` / `stop()` / `flush()` lifecycle
- `is_recording_enabled()` — env-gated on `MIRU_RECORD ∈ {1, true, yes, on}`
- `build_record(trace_dict, image_b64, question)` — privacy-stripped record: SHA-256 hex of source `image_b64` only, raw bytes never persisted; `overlay_b64` field stripped from the trace before serialisation
- `maybe_record(trace_dict, image_b64, question)` — fire-and-forget hook; swallows all errors so the request path is never broken by recorder failure
- `get_recorder()` / `reset_recorder()` — process-wide singleton with thread-safe init
- Storage backend: local filesystem when path has no URI scheme, `fsspec.open()` when scheme is present (`s3://`, `gs://`, `memory://`, …); `fsspec` is an optional `[storage]` extras install
- Per-batch file naming `traces-YYYYMMDDTHHMMSS-<microseconds>.jsonl` — uniform across cloud stores that don't support append (S3 et al.) and lexicographic time-sorted

#### API hooks (`miru/api/routes.py`, `miru/api/streaming.py`)
- `POST /analyze` — calls `maybe_record()` after building the trace
- `POST /analyze/stream` — calls `maybe_record()` inside `stream_analyze` after building the final trace; new `record: bool = False` parameter

#### CLI (`miru/cli/`)
- New entry point: `miru = "miru.cli:main"` registered in `[project.scripts]`
- `miru record list [--path <dir>]` — tab-separated `<records>\t<bytes>\t<path>` per file; prints `no recorded traces` for empty dirs
- `miru record export --out <file> [--path <dir>] [--format jsonl|csv]` — concatenate all recorded JSONL or flatten to CSV (`ts, question, image_sha256, answer, backend, latency_ms, n_steps`); skips corrupt JSON lines silently

#### Tests
- `tests/test_recorder.py` — 17 tests: hash determinism, privacy strip, ISO timestamp, env truthy/falsy gating, `maybe_record` no-op when disabled, JSONL line shape, flush count, stop drains queue, batching above `batch_size`, singleton identity, reset semantics, fsspec `memory://` round-trip, `/analyze` records, `/analyze` does not record when disabled, `/analyze/stream` records
- `tests/test_record_cli.py` — 8 tests: parser shape, empty-dir list, list output format, list main entrypoint, JSONL export concatenation, corrupt-line skip, CSV flattening, CSV main entrypoint

### Changed
- `miru/__init__.py`, `miru/config.py`, `pyproject.toml` — version bumped to `0.5.0`
- `pyproject.toml` — added `[project.scripts] miru = "miru.cli:main"`; new `[storage]` optional extras (`fsspec>=2024.2.0`); `fsspec` added to `[dev]`

### Privacy notes
- Stored records contain SHA-256 of the base64 image string and **never** the image itself or any derivative (overlay PNG is stripped before persistence)
- Hash covers the encoded payload byte-for-byte so identical uploads collide for de-duplication
- Question text is preserved verbatim (callers must scrub PII upstream if required)

### Test results
- 100 / 100 passing (4 skipped — gated CLIP real-backend tests)
- All 75 prior tests still pass

---

## [0.4.0] — 2026-05-01

### Added

#### Streaming protocol (`miru/api/streaming.py`)
- `stream_analyze(backend, image_array, question, …)` — async generator that drives `VLMBackend.stream_infer` and emits SSE-framed bytes
- Event grammar:
  - `event: step` — `{"step": <int>, "description": <str>, "confidence": <float>}` per reasoning step as it becomes available
  - `event: trace` — full `ReasoningTrace` JSON, schema-equivalent to the `/analyze` response (confidence, attention map, optional overlay)
  - `event: done` — empty payload sentinel
  - `event: error` — `{"error": <kind>, "detail": <str>}` on inference failure or timeout
- `: keepalive` SSE comments emitted at `keepalive_seconds` intervals so intermediate proxies do not idle-close long-running streams
- Per-request `timeout_seconds` budget (default 30s, query-tunable 1–300s); exceeding it emits a clean `error` event and closes the stream
- Producer/consumer pattern: synchronous `stream_infer` runs in a thread, marshaled to the event loop via `asyncio.Queue` (max 64) for backpressure

#### Backend interface (`miru/models/base.py`)
- `VLMStreamChunk` dataclass — `kind ∈ {"step", "final"}` with `step_index`, `step_description`, or full `output`
- `VLMBackend.stream_infer(image_array, question) -> Iterator[VLMStreamChunk]` — default impl replays `infer()` reasoning steps progressively; backends with native autoregressive token streaming should override

#### API endpoint (`miru/api/routes.py`)
- `POST /analyze/stream` — returns `text/event-stream`; query params `overlay: bool` and `timeout_seconds: float` (1–300, default 30)
- Same payload shape as `/analyze` (`ImageInput`); same fallback semantics for unknown backends; same image-decode safety
- Response headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`, `Connection: keep-alive`

#### Tracer helper (`miru/reasoning/tracer.py`)
- `step_confidence(base_confidence, step_index)` — extracted decay logic so the streaming path produces identical confidence values to the synchronous tracer

#### Tests (`tests/test_streaming.py`) — 10 new tests
- Endpoint emits steps + trace + done in order
- Step event shape (keys, types, confidence range)
- Streamed `trace` event matches `POST /analyze` response (modulo latency)
- `?overlay=true` populates `overlay_b64`
- Default `overlay_b64` is null
- Unknown backend falls back to mock
- `done` is the final event
- Step indices are sequential 1..N
- Default `stream_infer` replays steps then yields final
- Backend-side exception emits an `error` event without crashing the connection

### Changed
- `miru/config.py` — bumped `Settings.version` to `"0.4.0"`
- `pyproject.toml` — bumped project version to `"0.4.0"`
- `miru/__init__.py` — bumped `__version__` to `"0.4.0"`
- `tests/test_api.py` — version assertion updated

### Deviations from plan
- PLAN.md specified `GET /analyze/stream`; implemented as `POST` because the request payload includes a base64 image which does not fit a GET query string in any practical way. POST + `text/event-stream` is the canonical pattern for streaming responses with a non-trivial request body. This is *Konjo pushback* — the plan was a sketch, the right shape is POST.
- Did not introduce `sse-starlette` as a dependency; SSE framing is ~5 lines of byte concatenation. Honoring 건조 (strip to essence).

### Test results
- 75 / 75 passing (4 skipped — gated CLIP real-backend tests)
- Phase 1+2+3 unaffected (65/65 still pass)

---

## [0.3.0] — 2026-04-28

### Added

#### Visualization layer (`miru/visualization/`)
- `miru/visualization/overlay.py` — production-grade attention overlay utilities
  - `attention_to_heatmap(attention, colormap)` — converts 2-D float [0,1] array to (H,W,4) RGBA uint8; supports `"jet"`, `"hot"`, `"viridis"` colormaps implemented as piecewise-linear functions (zero matplotlib dependency)
  - `overlay_attention_on_image(image_rgba, attention, alpha)` — bilinearly upsamples attention to image spatial dimensions and alpha-blends heatmap over the base RGBA image; uses Pillow `BILINEAR` resize when available, nearest-neighbour NumPy fallback otherwise
  - `encode_png_b64(image_rgba)` — encodes (H,W,4) RGBA uint8 array to base64 PNG string; Pillow path when available, minimal pure-zlib PNG encoder (IHDR/IDAT/IEND) as fallback
  - `decode_image_b64(b64_str)` — decodes base64 image string (any Pillow-supported format) to RGBA uint8 array
  - `generate_overlay(image_b64, attention, alpha, colormap)` — end-to-end pipeline: decode → resize → heatmap → alpha-blend → encode PNG b64
- `miru/visualization/__init__.py` — module entry point; re-exports all five public functions

#### Schema update (`miru/schemas.py`)
- `ReasoningTrace.overlay_b64: str | None = None` — optional field carrying the base64-encoded PNG attention overlay; `None` when overlay was not requested or failed silently

#### Tracer update (`miru/reasoning/tracer.py`)
- `ReasoningTracer.trace()` — added `image_b64: str | None = None` and `generate_overlay: bool = False` parameters; when both are provided and true, calls `generate_overlay()` and attaches result to `overlay_b64`; any exception in overlay generation is silently suppressed so the trace always succeeds

#### API update (`miru/api/routes.py`)
- `POST /analyze?overlay=true` — new `overlay: bool = Query(default=False)` parameter; passes `image_b64` and `generate_overlay=True` to `ReasoningTracer.trace()` when enabled

#### Package update (`miru/__init__.py`)
- Exports `attention_to_heatmap` and `generate_overlay` at top-level
- Bumped `__version__` to `"0.3.0"`

#### Tests
- `tests/test_overlay.py` — 8 tests: zero-attention produces blue pixels, full-attention produces red pixels, dtype/range contract, overlay shape matches input, `encode_png_b64` returns valid base64, encode/decode round-trip preserves shape, `generate_overlay` pipeline with 1×1 white PNG, `ReasoningTrace.overlay_b64` defaults to `None`
- `tests/test_api_overlay.py` — 4 tests: `POST /analyze` without `overlay=true` returns `overlay_b64 == null`, with valid PNG + `overlay=true` returns non-empty `overlay_b64`, with invalid image + `overlay=true` does not crash, `GET /health` regression guard

### Changed
- `miru/config.py` — bumped `Settings.version` to `"0.3.0"`
- `pyproject.toml` — bumped project version to `"0.3.0"`
- `tests/test_api.py` — updated `test_health_version` assertion to `"0.3.0"`

---

## [0.2.0] — 2026-04-28

### Added

#### Backend registry (`miru/models/registry.py`)
- `register(name, factory)` — register a `Callable[[], VLMBackend]` under a string key
- `get(name)` — instantiate and return a backend by name; raises `KeyError` with helpful message if not found
- `available()` — return sorted list of registered backend names
- `register_defaults()` — register `"mock"` unconditionally; register `"clip"` when `transformers` is importable; idempotent (safe to call multiple times)

#### CLIP backend (`miru/models/clip.py`)
- `CLIPBackend` — `VLMBackend` subclass backed by `transformers.CLIPModel` + `CLIPProcessor`
- Lazy model loading: `_model` and `_processor` are `None` until the first `infer()` call
- Attention map extracted from the last ViT encoder layer's [CLS] token attention weights, averaged across heads, reshaped to `(grid_size, grid_size)` float32
- Answer is `"yes"` / `"no"` based on positive vs negated question text-image similarity
- Confidence mapped from cosine similarity range `[-1, 1]` → `[0, 1]`
- No module-level `torch` / `transformers` imports — imports confined to `_load()` and `infer()`

#### Routes update (`miru/api/routes.py`)
- Removed hardcoded `_backends` dict; replaced with `registry.register_defaults()` at module import
- `GET /health` now returns `registry.available()` — reflects dynamically registered backends
- `POST /analyze` uses `registry.get(payload.backend)` with `KeyError` fallback to default backend

#### Build (`pyproject.toml`)
- Added `[backends]` optional dependency group: `transformers>=4.35.0`, `torch>=2.0.0`, `Pillow>=9.0.0`

#### Tests
- `tests/test_registry.py` — 8 tests: register/get/available/defaults/idempotency/VLMBackend instance/name/health endpoint
- `tests/test_clip_backend.py` — 8 tests: 4 structural (no model load, always run) + 4 real-inference tests gated behind `MIRU_TEST_REAL_BACKENDS=1`

### Changed
- `miru/api/routes.py` — backend dispatch now uses the registry instead of a module-level dict; unknown backend names fall back to `settings.default_backend` via `KeyError` catch

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
