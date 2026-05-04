# PLAN.md — Miru Roadmap

**Project:** Miru — Multimodal Reasoning Tracer  
**Current version:** v0.6.0  
**Status:** Saliency benchmark harness complete, 129/129 tests passing (4 real-backend tests skipped without MIRU_TEST_REAL_BACKENDS=1)

---

## Phase 1 — Core Engine (v0.1.0) ✅ COMPLETE

**Goal:** Establish the full service skeleton with a deterministic mock backend, attention extraction, reasoning tracer, and FastAPI REST API.

**Delivered:**
- `miru/` Python package with layered architecture
- Abstract `VLMBackend` interface (`miru/models/base.py`)
- Deterministic `MockVLMBackend` with stable-hash Gaussian attention maps (`miru/models/mock.py`)
- `AttentionExtractor`: min-max normalization, block-average resize, top-k hotspot detection (`miru/attention/extractor.py`)
- `ReasoningTracer`: structured reasoning trace with decay confidence (`miru/reasoning/tracer.py`)
- FastAPI app with `GET /health` and `POST /analyze` endpoints (`miru/api/routes.py`)
- Pydantic v2 schemas: `ImageInput`, `AttentionMap`, `ReasoningStep`, `ReasoningTrace`, `HealthResponse`, `ErrorResponse`
- 30 passing tests across `test_models`, `test_attention`, `test_reasoning`, `test_api`
- GitHub Actions CI (Python 3.11, ubuntu-latest)
- `pyproject.toml` with `[dev]` extras

**Ship gate:** 30/30 tests pass, no placeholders, no TODOs.

---

## Phase 2 — Real VLM Backends (v0.2.0) ✅ COMPLETE

**Goal:** Wire in CLIP as an optional HuggingFace-backed backend and introduce a proper backend registry.

**Delivered:**
- `miru/models/registry.py` — `register()`, `get()`, `available()`, `register_defaults()` factory registry
- `miru/models/clip.py` — `CLIPBackend` using `transformers.CLIPModel` + `CLIPProcessor`; lazy-loads weights on first `infer()` call; derives 2-D attention map from last ViT encoder layer's [CLS] token attention averaged across heads
- `miru/api/routes.py` — rewired to use registry; `register_defaults()` called at module import; `/health` returns `registry.available()`; `/analyze` falls back to default backend on `KeyError`
- `pyproject.toml` — added `[backends]` optional dependency group (`transformers>=4.35.0`, `torch>=2.0.0`, `Pillow>=9.0.0`)
- `tests/test_registry.py` — 8 tests covering register/get/available/defaults/idempotency/health endpoint
- `tests/test_clip_backend.py` — 8 tests: 4 structural (no model load) + 4 gated behind `MIRU_TEST_REAL_BACKENDS=1`

**Ship gate:** 53/53 tests pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`; all 41 Phase 1 tests still pass.

---

## Phase 3 — Visualization (v0.3.0) ✅ COMPLETE

**Goal:** Return attention overlays as PNG/base64 alongside the JSON trace.

**Delivered:**
- `miru/visualization/overlay.py` — bilinear upsample attention map → RGBA heatmap overlay on input image; jet/hot/viridis colormaps implemented as piecewise-linear functions (no matplotlib); Pillow used when available, pure-zlib PNG encoder as fallback
- `miru/visualization/__init__.py` — exports `attention_to_heatmap`, `overlay_attention_on_image`, `encode_png_b64`, `decode_image_b64`, `generate_overlay`
- `ReasoningTrace.overlay_b64: str | None = None` — optional base64 PNG overlay field
- `ReasoningTracer.trace()` — extended with `image_b64` and `generate_overlay` optional params; silently falls back to `overlay_b64=None` on any overlay error
- `POST /analyze?overlay=true` — query parameter wires image payload → overlay pipeline → `overlay_b64` field in response
- `miru/__init__.py` — exports `attention_to_heatmap`, `generate_overlay`; bumped to v0.3.0
- `tests/test_overlay.py` — 8 tests: zero-attention blue, full-attention red, dtype/range, overlay shape, encode valid base64, round-trip decode, generate_overlay pipeline, schema field default
- `tests/test_api_overlay.py` — 4 tests: no-overlay null, overlay non-empty string, invalid-image no crash, health regression guard

**Ship gate:** 65/65 tests pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`; all 53 Phase 1+2 tests still pass.

---

## Phase 4 — Streaming (v0.4.0) ✅ COMPLETE

**Goal:** SSE streaming for token-by-token reasoning trace delivery.

**Delivered:**
- `miru/api/streaming.py` — `stream_analyze()` async generator, byte-level SSE framing (`step` / `trace` / `done` / `error` events), `: keepalive` comments, per-request `timeout_seconds` budget, backpressure via `asyncio.Queue(maxsize=64)` between thread-pool producer and event-loop consumer
- `miru/models/base.py` — `VLMStreamChunk` dataclass + `VLMBackend.stream_infer()` default generator that replays `infer()` reasoning steps progressively; native streaming backends override
- `miru/reasoning/tracer.py` — extracted `step_confidence()` so streamed and synchronous traces produce identical per-step confidence values
- `miru/api/routes.py` — `POST /analyze/stream` with `overlay: bool` and `timeout_seconds: float` (1–300) query params; same `ImageInput` payload, same unknown-backend fallback
- `tests/test_streaming.py` — 10 tests covering frame ordering, payload shape, parity with `/analyze`, overlay propagation, timeout budget, and backend-failure error events

**Deviations:**
- Endpoint is `POST`, not `GET` (per original sketch) — image_b64 doesn't fit a GET query string. POST + `text/event-stream` is the canonical pattern.
- No `sse-starlette` dependency — SSE framing is hand-rolled (5 lines). 건조.

**Ship gate:** 75/75 tests pass; all 65 prior tests still pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 5 — Dataset Recorder (v0.5.0) ✅ COMPLETE

**Goal:** Record reasoning traces to disk/S3 for fine-tuning data pipelines.

**Delivered:**
- `miru/recorder.py` — `TraceRecorder` threaded JSONL writer; per-batch timestamped files for fsspec compatibility (S3, GCS, memory, local); `MIRU_RECORD` env gating; `MIRU_RECORD_PATH` directory override
- Privacy: SHA-256 hex of `image_b64` is the only image-derived data persisted; raw bytes and overlay PNG are stripped before serialisation
- API hooks: `maybe_record()` called from `POST /analyze` and inside `stream_analyze` for `POST /analyze/stream`; errors swallowed so recording can never break the request path
- CLI (`miru/cli/`): `miru record list` (tab-separated record-count/size/path per file) and `miru record export --out <f> [--format jsonl|csv]`; entry point registered under `[project.scripts] miru`
- Optional `[storage]` extras install of `fsspec`; `fsspec` also bundled in `[dev]`
- 25 new tests across `tests/test_recorder.py` (17) and `tests/test_record_cli.py` (8); `memory://` round-trip exercises the fsspec path

**Ship gate:** 100/100 tests pass; all 75 prior tests still pass; 4 real-backend tests skip without `MIRU_TEST_REAL_BACKENDS=1`.

---

## Phase 6 — Saliency Benchmark Harness (v0.6.0) ✅ COMPLETE

**Goal:** Quantify how well an attention map tracks the salient region of an image, with a deterministic harness that runs in CI.

**Delivered:**
- `miru/bench/synth.py` — deterministic synthetic image + ground-truth-mask generator (single / two-blob / low-SNR variants), reproducible from `(seed, index)`
- `miru/bench/metrics.py` — `iou_at_topk_pct`, `auc_roc` (Mann-Whitney U with tie correction), `hit_at_k`; pure NumPy bilinear resampler
- `miru/bench/runner.py` — `run_benchmark()` drives any registered backend over the synth dataset, aggregates `{mean, std, p50, p95}` per metric, captures hardware metadata, persists JSON; `compare_results()` reports paired delta + paired-t statistic
- `miru/cli/bench.py` — `miru bench run / show / compare` subcommands wired into the existing CLI entry point
- `benchmarks/results/baseline-mock.json` — first locked-in baseline (n=30, seed=42): IoU 0.062, AUC 0.627, hit@1 0.100, latency 0.080 ms — confirms the mock backend's attention is question-hash-driven, not image-driven
- 29 new tests across `tests/test_bench.py`

**Konjo deviation from sketch:** Original plan called for a held-out VQA slice; shipped a synthetic harness instead because an external dataset adds a download dependency, license fragility, and runtime flakiness for a deterministic check. Synthetic blobs with known ground truth deliver the same statistical claim, license-clean, in seconds, with zero new deps. Extensible — a future PR can plug VQA-X behind the same metric interface without touching consumers.

**Ship gate:** 129/129 tests pass; all 100 prior tests still pass; first real benchmark recorded against the mock backend.

---

## Phase 7 — TBD

Open candidates (Discovery to refine before sprinting):
- Score the CLIP backend against the new harness and publish a `clip-vs-mock` comparison artefact — concrete proof that attention quality varies meaningfully across backends, and a regression gate for future CLIP changes.
- Native VLM streaming backend (LLaVA / Idefics / Qwen-VL with token-level attention) so `/analyze/stream` produces genuinely incremental reasoning instead of replaying a single-shot inference.
- Real-image benchmark slice: plug VQA-X or COCO-Saliency behind the existing metric interface and publish the curve alongside the synthetic baseline.
- gRPC alternative to the FastAPI surface for in-cluster low-latency inference.
