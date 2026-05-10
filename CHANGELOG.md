# CHANGELOG

All notable changes to Miru are documented here.  
Format: [Conventional Commits](https://www.conventionalcommits.org/) + [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased] — Phase 13: LIME, GradCAM, side-by-side compare, eye UI

### Added

#### Two new explainers
- `miru/lime_explainer.py` — LIME (Ribeiro et al. 2016) for image inputs.
  Pure-NumPy grid-based superpixel segmentation, mean-colour occlusion of
  each superpixel, weighted-least-squares surrogate solved with
  `np.linalg.lstsq`, no scikit-learn dependency. Deterministic under a seed.
- `miru/gradcam_explainer.py` — occlusion-sensitivity saliency (Zeiler &
  Fergus 2014). The gradient-free cousin of true Grad-CAM, exposed under
  the `gradcam` name with an explicit docstring noting that real
  backprop-based Grad-CAM requires a torch/CNN backend (future work).

#### API
- `POST /explain/compare` — runs two methods on one image and returns both
  base64 PNG overlays + saliency grids + top-region rings. Used by the
  visual demo for side-by-side comparison.
- `/explain` now dispatches on `method ∈ {attention, lime, gradcam}` —
  all three return the same response shape so clients are method-agnostic.
- `/methods` reports lime + gradcam as `implemented`; only `shap` remains
  on the roadmap.
- New bounded request fields: `n_samples ≤ 256`, `n_segments ≤ 144`,
  `occlusion_grid ≤ 16` — keeps a public deploy from being made to do
  unbounded `backend.infer()` calls per request.

#### Visual demo — `demo/visual.html`
- A single large stylized eye (pure CSS/SVG) on a deep `#06060f` field.
  Iris in Konjo purple `#7c3aed` with two independently-rotating ring
  layers (28s and 16s, opposite directions) and a 4.5s breathing pulse.
- Image loads **into** the iris — the pupil swells from 130px → 200px
  and the source image fills it.
- Heatmap bleeds outward from the pupil with `mix-blend-mode: screen` —
  the eye is literally focusing its attention on the image.
- Three method icons: gradient (gradcam), mosaic (lime), waveform
  (attention). Active one glows.
- Top-3 regions appear as numbered amber focus rings around their
  hotspot positions inside the eye, like an iris dilating around
  regions of interest.
- "split view" toggle — the single eye divides into two side-by-side
  eyes via CSS layout transition; left shows attention, right shows
  the chosen comparison method, both populated by `POST /explain/compare`.
- Three procedurally-generated 64×64 sample images shown as iris-thumbnail
  circles below — click to load.
- Eye blinks once on load (`@keyframes blink`), pupil pulses while idle
  (`@keyframes pupil-pulse`).

#### Tests
- `tests/test_explainers.py` — 8 tests covering LIME segmentation, LIME
  determinism under seed, LIME saliency normalization & shape, GradCAM
  shape & call-count, plus all input-validation rejections.
- `api/test_api.py` — 7 new tests for `/explain` parametrized over each
  implemented method, `/explain/compare` happy path + same-method-rejection
  + unknown-method-rejection, and a guard on `/methods` reporting lime
  and gradcam as `implemented`.

### Changed
- `IMPLEMENTED_METHODS = ("attention", "lime", "gradcam")` (was `("attention",)`).
- `ROADMAP_METHODS = ("shap",)` (was `("gradcam", "lime", "shap")`).
- The `roadmap-method-returns-400` test now picks the first roadmap
  method dynamically so it stays green as methods are promoted.

### Notes
- 252/252 pass (15 new, 237 existing); 4 real-backend tests still skip
  without `MIRU_TEST_REAL_BACKENDS=1`.
- Naming honesty: `gradcam` ships as occlusion-sensitivity (a real,
  citable saliency method common in XAI toolkits). The `/methods`
  description and `miru/gradcam_explainer.py` docstring say so explicitly.

---

## [Unreleased] — Phase 10: deployable REST API

### Added

#### `api/` — deployable explainability surface
- `api/main.py` — FastAPI app with five endpoints:
  - `GET  /health`     — status, version, registered backends, implemented methods
  - `GET  /methods`    — explanation methods (implemented + roadmap) and registered models
  - `POST /explain`    — saliency/attention map for one (image, model, method); returns
    base64 PNG overlay, attention grid, top-k regions, latency
  - `POST /benchmark`  — drives `miru.bench.runner.run_benchmark` over the synth GT-mask
    harness; returns aggregate IoU / AUC-ROC / hit@1 / latency stats (mean, std, p50, p95)
  - `POST /compare`    — paired comparison of two backends via `compare_backends`; returns
    per-metric stats for each side, paired delta, paired-t statistic, and a winner verdict
- `method` field is honest about scope: only `attention` is implemented; `gradcam | lime |
  shap` are listed as roadmap and rejected with **400 + clear message** rather than silently
  falling back to attention extraction
- `n` capped at 100, `size` capped at 128 — bounded compute on a public deploy
- CORS middleware open by default for browser clients (dashboard / playgrounds)

#### Deployment
- `api/requirements.txt` — runtime deps (fastapi, uvicorn, pydantic, numpy, Pillow)
- `api/Dockerfile` — slim Python 3.11 image, non-root user, `$PORT` honoured
- `render.yaml` — Render.com web service manifest pointing at `api/Dockerfile`

#### Tests
- `api/test_api.py` — 13 tests covering: health, methods listing, explain happy path with
  a real synthetic 16×16 PNG, malformed-image / unknown-model / roadmap-method / unknown-method
  400 contracts, benchmark aggregation shape and `n`-cap rejection, `mock`-vs-`mock` compare
  is a perfect tie

### Notes
- Distinct from the in-package `miru/api/` router (the dev server) — `api/` is the
  deployable artefact and depends on the `miru` package as a library.
- 237 tests pass (13 new in `api/`, 224 existing); 4 real-backend tests still skip without
  `MIRU_TEST_REAL_BACKENDS=1`.

---

## [1.1.0] — 2026-05-09

### Added

#### Grad-CAM explainer (`miru/gradcam.py`)
- `compute_gradcam(activations, gradients) -> np.ndarray` — pure-NumPy core.
  Implements Selvaraju et al., 2017: `α_k^c = mean_{i,j} ∂y^c/∂A^k_{ij}`,
  heatmap `L^c = ReLU(Σ_k α_k^c · A^k)`, then min-max to `[0, 1]`. Degenerate
  (all-negative-evidence) maps return all-zero instead of dividing by ~0.
- `attention_to_cam(attention) -> np.ndarray` — fallback for ViT backbones.
  Accepts `(H, W)` or `(heads, seq, seq)`; for the latter, averages heads,
  drops the [CLS] row, and reshapes to a square patch grid.
- `top_k_regions(heatmap, k) -> [(row, col, score), ...]` — argpartition-based
  top-k extractor, sorted score-desc.
- `GradCAMExplainer(model, target_layer=None)` — torch-aware explainer.
  Auto-finds the last `Conv2d` via `model.modules()`; if none exist (pure ViT
  case), sets `uses_attention_fallback=True` and uses
  `output.attentions[-1]` from a forward pass.
- Forward + full-backward hooks capture activations and gradients without
  retain_graph; `torch` is imported lazily inside the hook path so the module
  loads cleanly without torch installed.
- `GradCAMResult(heatmap, top_regions, target_class, used_fallback)` — frozen
  dataclass returned by every entry point.

#### `POST /explain` endpoint (`miru/api/routes.py`)
- `method: gradcam` — **implemented** (M11 ship gate). Falls back to the
  attention-weight method when the active backend has no Conv2d layers.
- `method: attention` — implemented (raw VLM attention).
- `method: lime` / `method: shap` — return `501` with status `roadmap`.
- `method: <unknown>` — returns `422`.
- `top_regions` carry normalised image-relative bboxes (`bbox_x1..y2 ∈ [0, 1]`)
  so demo callers can scale them against the rendered image without knowing
  the heatmap resolution.
- `?overlay=true` query param wires the heatmap through
  `miru.visualization.overlay.generate_overlay` for a base64 PNG.
- `EXPLAIN_METHODS: dict[str, str]` exported as the canonical method registry.

#### Visual demo (`demo/visual.html`)
- Interactive single-page explorer for `/explain`. Dark theme matching
  `demo/index.html` design tokens.
- Three bundled procedural sample images (`demo/sample_images/blob.png`,
  `two_blobs.png`, `gradient.png`) generated with the in-repo pure-zlib PNG
  encoder — no external assets, no Pillow runtime dependency for CI.
- Method selector (attention | gradcam), backend selector, free-form question.
- Side-by-side: original image with SVG bounding-box overlay vs. heatmap
  overlay PNG.  When the API can't render the overlay (no Pillow), the page
  falls back to a client-side jet-colormap canvas render of the raw heatmap.
- Top-5 attended regions table with score bars and (row, col) +
  (x1, y1, x2, y2) bbox coords.
- Posts to same-origin `/explain?overlay=true` by default; configurable via
  the `API_BASE` constant at the top of the script.

#### Tests (`tests/test_gradcam.py`) — 22 new tests
- Pure-numpy core: shape/dtype, normalisation range, ReLU kills negative
  evidence, single-channel hot-spot localisation, shape-mismatch raise,
  reject-non-3-D.
- Attention fallback: 2-D pass-through, multi-head collapse to (7, 7) for
  `(12, 50, 50)`, uniform-input zeros.
- `top_k_regions` ordering and `k <= 0` edge.
- `GradCAMExplainer.from_arrays` returns the dataclass; `from_attention`
  marks `used_fallback=True`; `explain()` without a model raises;
  `_find_last_conv` returns `None` for objects without `.modules()`.
- `/explain` endpoint: attention 200, **gradcam 200** (M11 gate),
  bbox normalised to `[0, 1]`, lime 501, unknown 422, overlay query string,
  unknown-backend fallback to default.

### Changed
- `miru/__init__.py`, `miru/config.py`, `pyproject.toml` — version 1.0.0 → 1.1.0.
- `tests/test_api.py` — `/health` version assertion updated to 1.1.0
  (previously stuck at 0.7.0; pre-existing drift).
- `miru/__init__.py` — re-exports `GradCAMExplainer`, `GradCAMResult`,
  `compute_gradcam`.
- `miru/schemas.py` — adds `ExplainRequest`, `ExplainResponse`, `ExplainRegion`.

### Test results
- 246 / 246 passing (4 skipped — gated CLIP real-backend tests)
- All 224 prior tests still pass

---

## [0.7.0] — 2026-05-05

### Added

#### Attention-map export (`miru/bench/export.py`)
- `generate_report(result, out_dir, …)` — takes a saved benchmark JSON, re-generates
  all synthetic images deterministically from `(seed, index)`, composites the attention
  heatmap on each image, draws a yellow ground-truth mask border, and writes:
  - `report.html` — self-contained HTML page with inline base64 thumbnails, aggregate
    metric tables, and per-sample IoU/AUC/hit@1/latency tiles
  - `sample_NNN_overlay.png` / `sample_NNN_raw.png` — per-sample PNG tile pairs (optional)
- `render_sample(sample_rec, bench_seed, bench_size, …)` — re-generates one synthetic
  image, runs the mock backend for a deterministic attention map, composites it.
  Returns `(raw_rgba, overlay_rgba)` as `(H, W, 4)` uint8 RGBA pairs.
- `_composite_overlay(image, attn_grid, alpha, colormap)` — bilinear upsample
  (`miru.bench.metrics.bilinear_upsample`, `align_corners=True`) then alpha-blend.
  Math: `out = clip(heatmap * α + base * (1-α), 0, 255)`.
- `_mask_border_rgba(mask)` — 4-connected erosion to extract boundary pixels, rendered
  as semi-transparent yellow (R=255, G=220, B=0, A=180).
- `_alpha_composite(bottom, top)` — Porter-Duff over compositing in float32 with
  correct alpha channel propagation: `out_A = α_t + α_b * (1-α_t)`.
- Zero new runtime dependencies: reuses `miru.bench.metrics.bilinear_upsample`,
  `miru.visualization.overlay.{attention_to_heatmap,encode_png_b64}`, and
  `miru.bench.synth.generate_sample`.

#### CLI (`miru/cli/export.py`, `miru/cli/__init__.py`)
- `miru export <result.json> <out_dir>` — top-level subcommand (not nested under `bench`)
- Flags: `--alpha 0.5`, `--colormap jet|hot|viridis`, `--no-mask-border`, `--no-png-tiles`
- Returns exit 0 on success, 1 with a clear error message on load failure

#### Tests (`tests/test_export.py`) — 32 new tests
- `_image_to_rgba`: shape/dtype, alpha=255, value clipping
- `_composite_overlay`: shape/dtype, all three colormaps, alpha=0 identity
- `_mask_border_rgba`: shape/dtype, interior transparent, edge opaque, empty mask
- `_alpha_composite`: shape/dtype, transparent-top identity, opaque-top dominates
- `render_sample`: shape/dtype contracts, determinism, mask-border on/off,
  different indices produce different images
- `generate_report`: HTML created, aggregate metrics present, inline images embedded,
  PNG tiles count, no-tiles mode, valid PNG magic bytes, all colormaps, return type,
  creates nested directories
- CLI: parser accepts `export` subcommand + all flags, happy-path end-to-end,
  `--no-png-tiles` suppresses PNGs, bad-result-path returns exit 1 with error message,
  HTML content correctness (n, seed, backend in output)

### Fixed
- `pyproject.toml` `[dev]` extras were missing `Pillow>=9.0.0`; this caused a
  pre-existing failure in `test_analyze_with_overlay_returns_nonempty_string` when
  running in a fresh venv. Added `Pillow>=9.0.0` to `[dev]`.

### Changed
- `miru/__init__.py`, `miru/config.py`, `pyproject.toml` — version 0.6.0 → 0.7.0
- `miru/cli/__init__.py` — `export` top-level subcommand wired in; module docstring updated

### Test results
- 161 / 161 passing (4 skipped — gated CLIP real-backend tests)
- All 129 prior tests still pass

---

## [0.6.0] — 2026-05-04

### Added

#### Saliency benchmark harness (`miru/bench/`)
- `miru/bench/synth.py` — deterministic synthetic image + ground-truth-mask generator with three difficulty variants:
  - `single` — one bright Gaussian blob on smooth coloured noise
  - `two` — two well-separated blobs (centroids guaranteed `> 4σ` apart, with a deterministic fallback if the rejection sampler can't find a clean pair)
  - `low_snr` — single blob with reduced amplitude over stronger noise
  Every sample is fully reproducible from `(seed, index)`; ground-truth mask is the union of disks of radius `1.6σ` centred at each blob.
- `miru/bench/metrics.py` — three saliency metrics, all pure NumPy:
  - `iou_at_topk_pct(attn, mask, top_pct)` — bilinearly upsample attention, threshold at the top `top_pct`, IoU vs mask
  - `auc_roc(attn, mask)` — pixel-level AUC via Mann-Whitney U (with tie correction). Returns chance level (0.5) on degenerate masks rather than raising
  - `hit_at_k(attn, mask, k)` — fraction of top-k attention pixels inside the mask; downsamples the mask onto the attention grid (cheaper than upsampling attn)
  - `bilinear_upsample` — `align_corners=True` 2-D resampler used by the metrics
- `miru/bench/runner.py` — `run_benchmark(backend, n, seed, …)` drives any registered VLMBackend over a synth dataset, scores each sample, aggregates `{mean, std, p50, p95, n}` per metric, and persists a single JSON document with hardware metadata, schema version, and per-sample drilldown. `compare_results(a, b, metric)` enforces paired runs (same n + seed) and reports mean delta + paired t-statistic + degrees of freedom (no SciPy dep — caller can compute the p-value if needed).

#### CLI (`miru/cli/bench.py`)
- `miru bench run --backend <name> --n N --seed S [--out PATH] [--top-pct 0.20] [--k 1]` — execute and print summary; optionally save JSON
- `miru bench show <result.json>` — pretty-print a saved run with per-variant IoU breakdown
- `miru bench compare <a.json> <b.json> [--metric iou|auc|hit1|latency_ms]` — paired delta, "→ b WINS / a WINS / tie" verdict
- All three subcommands wired into the existing `miru` entry point

#### First baseline result
- `benchmarks/results/baseline-mock.json` — n=30, seed=42 against the mock backend. Aggregate: IoU **0.062**, AUC **0.627**, hit@1 **0.100**, latency **0.080 ms**. The harness immediately confirms what the mock's design implies: its attention is question-hash-driven and only weakly related to image content. This is the floor against which real backends (CLIP, future VLMs) will be measured.

#### Tests (`tests/test_bench.py`) — 29 new tests
- Synth: shape/dtype contracts, determinism on `(seed, index)`, distinct outputs for different indices, fixed variant cycle, two-variant has two centroids, mask centroid matches recorded centroid (within 1.5px), `generate_dataset` size
- Metrics: bilinear upsample identity + corner preservation, IoU perfect / disjoint / `top_pct` validation, AUC perfect / inverted / random / degenerate-mask chance, hit@k inside / outside / mask-resampling / `k≥1` validation
- Runner: smoke shape contract, unknown-backend fallback to mock, all metrics in `[0,1]` and latency `> 0`, hardware metadata captured, save→load round-trip, `compare_results` zero-delta on identical seeds, `compare_results` rejects unpaired seed/n
- CLI: parser accepts all three subcommands, run writes JSON, show round-trips through main entry point, compare prints "tie" on identical seeds

### Changed
- `miru/__init__.py`, `miru/config.py`, `pyproject.toml`, `tests/test_api.py` — version 0.5.0 → 0.6.0
- `miru/cli/__init__.py` — `bench` subcommand wired into the parser; module docstring updated

### Deviations from plan
- PLAN.md sketched Phase 6 as "trained-saliency benchmarks: take a held-out VQA slice". Shipped as a self-contained synthetic harness instead. Rationale: an external dataset adds a download dependency, license fragility, and runtime flakiness for what should be a deterministic CI check. Synthetic blobs with known ground truth deliver the same statistical claim, license-clean, in seconds, with zero new deps. 건조 — strip to essence. The harness is also extensible: a future PR can plug VQA-X behind the same `iou_at_topk_pct` / `auc_roc` / `hit_at_k` interface without touching downstream consumers.

### Test results
- 129 / 129 passing (4 skipped — gated CLIP real-backend tests)
- All 100 prior tests still pass

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
