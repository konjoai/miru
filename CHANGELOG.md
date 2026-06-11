# CHANGELOG

All notable changes to Miru are documented here.  
Format: [Conventional Commits](https://www.conventionalcommits.org/) + [Keep a Changelog](https://keepachangelog.com/).

---

## [1.7.0] — Phase 24: ROI-targeted explanation

### Added

#### `BoundingBox` model (`api/main.py`)
- New Pydantic model with relative `[0, 1]` coordinates (`x1`, `y1`, `x2`, `y2`).
  A `model_validator` enforces `x2 > x1` and `y2 > y1` at construction time.

#### `roi` field on `ExplainRequest`
- Optional `roi: BoundingBox | None = None` added to `POST /explain`.
  When set, saliency is computed only on the cropped region and embedded back
  into a full-resolution zero grid. The VLM answer/confidence always come from
  the full image. Works for all four methods: `attention`, `lime`, `gradcam`,
  `shap`. Returns HTTP 400 when the crop maps below 4×4 pixels.

#### Cache-key isolation
- `_explain_cache_key` includes the roi coordinates, so different sub-regions
  on the same image never share a cached result.

### Tests
- `api/test_roi.py` — 13 HTTP tests covering happy path, all methods, zero-
  outside-bbox invariant, validation rejections, and cache key isolation.
- `tests/test_roi.py` — 6 unit tests for `BoundingBox` model validation.

---

## [1.6.0] — Phase 23: input sensitivity / robustness

### Added

#### `miru/sensitivity.py` — explainer-agnostic robustness probe
- `compute_sensitivity(saliency_fn, image, *, baseline_grid, baseline_answer,
  sigmas, n_trials, seed, stability_threshold)` — sweeps seeded Gaussian noise
  at each σ, re-runs `saliency_fn` `n_trials` times per σ, and measures mean
  absolute attribution drift of the normalised saliency from the clean
  baseline. Returns `SensitivityResult`: `stability_score = 1 − mean_drift`
  (clamped to `[0, 1]`), `worst_sigma`, `worst_drift`, `is_stable`, and
  per-σ `PerturbationResult`s.
- Method-agnostic by design — it takes a `saliency_fn` (image → 2-D grid), so
  it covers `attention` / `lime` / `gradcam` / `shap` uniformly.
- `baseline_grid` short-circuit lets callers pass the clean-image saliency they
  already computed, avoiding a redundant (and for LIME/SHAP, expensive)
  explainer run.
- `attribution_drift(a, b)` — mean absolute per-cell difference; raises on a
  shape mismatch. Fully deterministic under `seed` (cf. Ghorbani et al. 2019,
  "Interpretation of Neural Networks Is Fragile").

#### `POST /explain/sensitivity` (`api/main.py`)
- `SensitivityRequest` carries the same explainer knobs as `/explain` plus
  `sigmas` (≤ 8, each in (0, 1]), `n_trials` (≤ 8 — bounds `backend.infer`
  fan-out), `seed`, `stability_threshold`.
- Drives all four methods through the existing `_run_method` dispatch, so the
  robustness measured is exactly the map `/explain` would return.
- `SensitivityResponse`: `model_name`, `method`, `baseline_answer`,
  `stability_score`, `is_stable`, `worst_sigma`, `worst_drift`, `per_sigma[]`,
  `latency_ms`.
- 400 on out-of-range / empty / oversized `sigmas`, unknown method, unknown
  model, or an undecodable image.

#### Tests
- `tests/test_sensitivity.py` (12 unit) + `api/test_sensitivity.py` (12 HTTP) —
  24 new tests.

### Changed
- Version bumped to `1.6.0`. `tests/test_api.py` health-version assertion → `1.6.0`.
- `miru/__init__.py` exports `compute_sensitivity` and `SensitivityResult`.

### Notes
- 639 / 639 passing (24 new, 615 existing); 5 skipped.
- **Scope discipline:** the exploratory build of this feature also reintroduced
  a history store, model comparison, and pattern search — all of which already
  shipped in Phases 20 and 22 (`miru/history.py`, `miru/model_comparison.py`,
  `miru/search.py`). Those duplicates were dropped; only the novel robustness
  probe was kept and wired into the existing explainer dispatch.

---

## [Unreleased] — Phase 22: model comparison · post-hoc consensus · search

### Added

#### `miru/model_comparison.py` — per-model aggregation
- `compare_models(models, *, limit, method)` pulls per-model
  history slices and computes `n_records`, `mean_confidence`,
  `mean_latency_ms`, `mean_fidelity`, `n_with_fidelity`, `ece` (via
  `compute_calibration`), and `method_distribution` for each model.
- Three winner verdicts: by mean confidence (higher wins), by mean
  fidelity (higher wins), by ECE (lower wins). Each winner is
  `None` when no model has data for that metric. Ties resolve in
  input order.
- Validates: non-empty / distinct `models`; `limit ∈ 1..200`.

#### `miru/posthoc_consensus.py` — weighted consensus from analysis_ids
- `build_consensus(records, *, weighting, top_k)` aligns all
  attention grids (bilinear-upsample to the max shape) and computes
  a weighted average. Three weighting modes:
  - `fidelity` (default): per-record weight = `fidelity_score`;
    records without fidelity get the population floor; all-missing
    falls back to uniform.
  - `confidence`: per-record weight = `confidence`; all-zero falls
    back to uniform.
  - `uniform`: every record gets weight 1.0.
- Each contributing record receives an `agreement_score ∈ [-1, 1]`
  (cosine between its grid and the consensus).
- Distinct from `miru/consensus.py` (Phase 13) which runs methods
  live; this combines records that already ran.

#### `miru/search.py` — pattern search over history
- `search_by_pattern(*, query_grid | query_analysis_id, method,
  model, top_k, min_similarity, max_scan)` finds historical
  analyses by cosine similarity of their attention grids.
- Bilinearly aligns candidate grids to the query's shape, so
  methods producing different resolutions are still comparable.
- Self-exclusion when querying by `analysis_id`. Malformed candidate
  records (no grid) are silently skipped.
- Exact (no embedding index) — fine at audit-log scale; swap in
  faiss/hnswlib behind the same interface when corpus grows.

#### `GET /explain/models/compare`
- Query: `models` (comma-separated, distinct, ≤ 8), `limit ≤ 200`,
  optional `method` filter.
- Returns per-model `ModelStatsBlock` + three winner verdicts + the
  echoed `filter_method` and `limit`.

#### `POST /explain/consensus/by_ids`
- Body: `{analysis_ids[2..16], weighting, top_k}`.
- Returns the consensus grid, per-record contributions (weight +
  agreement_score), top-K consensus regions, echoed weighting,
  grid dimensions.
- 400 on duplicate IDs / bad weighting; 404 on missing record.

#### `POST /explain/search`
- Body: either `query_grid` or `query_analysis_id` (exclusive),
  optional `method` / `model` filters, `top_k ≤ 50`,
  `min_similarity ∈ [-1, 1]`, `max_scan ≤ 2000`.
- Returns `SearchMatch[]` (analysis_id, ts, method, backend,
  question, similarity), `n_candidates`, `n_scanned`, echoed
  `top_k`, echoed `query_analysis_id`.
- 400 on bad arguments; 404 on missing `query_analysis_id`.

### Tests
- `tests/test_model_comparison.py` — 12 unit tests
- `tests/test_posthoc_consensus.py` — 16 unit tests
- `tests/test_search.py` — 18 unit tests
- `api/test_phase22_endpoints.py` — 20 HTTP tests
- Full suite: **615 / 615 passing** (549 baseline → +66); 5 skipped.

### Notes
- All three endpoints are read-side: they never invoke a backend or
  write a record. They compose around the recorded explanation
  store opened up by Phase 20's `query_records()` /
  `find_record_by_id()`.

---

## [Unreleased] — Phase 21: scale-space attention ensemble (v1.5.0)

### Added

#### `miru/ensemble.py` — multi-scale attention aggregation
- `AttentionEnsemble(scales, weights, extractor)` — runs `backend.infer()` at
  each scale, normalises each attention map via `AttentionExtractor`, and
  produces a weighted average re-normalised to `[0, 1]`.
- `EnsembleResult` frozen dataclass: `ensemble_grid`, `per_scale` list,
  `scales_used`, `scales_skipped`, `grid_h`, `grid_w`.
- `_bilinear_resize_image(image, scale)` — pure-NumPy bilinear image resize;
  returns `None` when either dimension falls below `MIN_DIM (4px)`.
- `DEFAULT_SCALES = (0.5, 1.0, 1.5)` exported as module constant.
- All-fail fallback: returns all-zero grid + warning instead of raising.

#### `POST /explain/ensemble` — `api/main.py`
- `EnsembleRequest`: `image_b64`, `model_name`, `question`, `scales` (1–5,
  each in `(0, 4]`), optional `weights`, standard `alpha`/`colormap`/`top_k`.
- `EnsembleResponse`: `ensemble_grid`, `per_scale` (scale + grid per entry),
  `scales_used`, `scales_skipped`, `top_regions`, `overlay_b64`, `latency_ms`.
- 400 on unknown model, bad image, out-of-range scale, weight-length mismatch.

### Tests
- `tests/test_ensemble.py` — 27 tests: resize helper contracts, unit ensemble
  (single-scale, value range, scale tracking, custom weights, error cases, all-
  fail zeros, resolution forwarding), API (happy path, response fields, grid
  values, per_scale count, all error contracts, echo, health regression).

### Changed
- Version bumped to `1.5.0`.
- `tests/test_api.py` health-version assertion updated to `1.5.0`.

### Test results
- **497 / 497 passing**, 5 skipped.

---

## [Unreleased] — Phase 20: history · calibration · diff

### Added

#### `miru/history.py` — query + calibration core
- `query_records(...)` — filtered + paginated newest-first listing.
  Filters: `method`, `model`, `min_confidence`, `since` (ISO-8601).
  Pagination: `limit ∈ 1..200`, `offset ≥ 0`.  Drains the singleton
  recorder before scanning so records produced microseconds ago by
  `maybe_record` are visible (same fix `find_record_by_id` uses).
- `compute_calibration(records, n_bins)` — Expected Calibration Error
  + per-bin reliability curve.  ECE = Σ (n_b/N) × |conf_b − fid_b|.
  Skips records without a fidelity score; clamps out-of-range values;
  validates `n_bins ∈ 2..50`; empty population returns `ece=0.0`.

#### `miru/diff.py` — post-hoc diff of two analysis records
- `diff_records(rec_a, rec_b, top_n)` — aligns attention grids,
  computes cosine similarity (on raw vectors, sign/magnitude
  preserved), L2 distance (on min-max normalised grids), signed delta
  grid, top-N changed cells, and a human-readable summary using a 3×3
  spatial grid ("A focused more on the bottom-left; B shifted toward
  the top-right").

#### `POST /explain/diff`
- Accepts two `analysis_id`s.  Returns `DiffResponse` with cosine,
  L2, delta grid, top changed cells, and summary string.
- 400 on identical IDs.  404 on missing record.

#### `GET /explain/history`
- Query: `limit ∈ 1..200`, `offset ≥ 0`, `method`, `model`,
  `min_confidence ∈ [0, 1]`, `since` (ISO-8601).
- Returns paginated `HistoryItem[]` + `total`.  Stripped of bulky
  `attention_grid` / `top_regions`; fetch the full payload via
  the existing `/analysis/{id}/export?format=json`.

#### `GET /explain/calibration`
- Query: `n_bins ∈ 2..50`, `method`, `model`, `limit ≤ 200`.
- Returns ECE scalar + bins + filter echo.
- Empty / no-fidelity population returns `n=0, ece=0.0` (clients
  render an "insufficient data" state).

### Tests
- `tests/test_history.py` — 22 unit tests
- `tests/test_diff.py` — 13 unit tests
- `api/test_history_diff_calibration.py` — 17 HTTP tests
- Combined with main's Phase 19 (470 baseline) → 491 / 491 passing total once
  rebased; +52 new vs. local 387 baseline.

### Notes
- Export (`/analysis/{id}/export?format=…`) was already shipped in
  Phase 14 — not re-built.
- The three new endpoints compose naturally: history is the
  foundation; calibration is one aggregation over the filtered view;
  diff is a 2-record operation that reuses `find_record_by_id`.
- This work was renumbered from "Phase 17" → "Phase 20" during rebase
  because main shipped Phases 17 (cross-modal), 18 (annotation), and
  19 (dataset analytics) in parallel.

---

## [Unreleased] — Phase 19: dataset-level saliency analytics (v1.4.0)

### Added

#### `miru/dataset_analytics.py` — batch saliency aggregation
- `aggregate_saliency(grids)` — cell-wise mean + std over a list of 2-D
  saliency grids; bilinearly resamples to the first grid's shape.
- `detect_spurious(mean_grid, std_grid, *, mean_threshold, cv_threshold,
  n_samples)` — flags cells where mean ≥ threshold AND CV (std/mean) <
  threshold.  Suppressed when n < 3 (variance unreliable from few samples).
  Returns ``(bool_mask, [(row, col), ...])`` sorted by descending mean.
- `analyse_dataset(grids, ...)` — full pipeline; returns `DatasetAnalytics`
  frozen dataclass: `mean_grid`, `std_grid`, `cv_grid`, `spurious_mask`,
  `spurious_cells`, `n_samples`, `grid_h`, `grid_w`.
- `SPURIOUS_MEAN_THRESHOLD = 0.5`, `SPURIOUS_CV_THRESHOLD = 0.5`,
  `MIN_SAMPLES_FOR_SPURIOUS = 3` exposed as module constants.

#### `POST /analyze/batch` — `api/main.py`
- `DatasetAnalyticsRequest`: `images` (1..64 `DatasetBatchItem`), `model_name`,
  `method`, `mean_threshold`, `cv_threshold`, and all per-method tuning knobs.
- `DatasetAnalyticsResponse`: `mean_grid`, `std_grid`, `cv_grid`,
  `spurious_cells` (with `mean_saliency` per cell), `per_image` (index, answer,
  confidence, attention_grid), `n_images`, `latency_ms`.
- 400 on unknown model/method or bad image; 422 on empty images list.

### Tests
- `tests/test_dataset_analytics.py` — 29 tests: unit (aggregate contracts,
  spurious flag all four quadrant cases, sort order, zero-mean guard,
  analyse_dataset shape/range/std-zero/all-spurious/empty-error) + API
  (happy path, response shape, index order, mean-grid values in range,
  spurious list present, single-image no-spurious, model echo, error contracts,
  health regression).

### Changed
- Version bumped to `1.4.0`.
- `tests/test_api.py` health-version assertion updated to `1.4.0`.

### Test results
- **470 / 470 passing**, 5 skipped.

---

## [Unreleased] — Phase 18: expert annotation alignment (v1.3.0)

### Added

#### `miru/annotation.py` — alignment scoring
- `compare_annotation(saliency, mask, *, answer_correct, top_pct)` —
  scores a saliency grid against a binary ground-truth mask supplied by a
  human annotator.
- `AnnotationAlignment` frozen dataclass: `iou`, `auc_roc`, `spearman_r`,
  `top_pct`, `misaligned`.
- Reuses `iou_at_topk_pct` and `auc_roc` from `miru.bench.metrics`.
- `_spearman` + `_rank` — pure-NumPy Spearman rank correlation; no SciPy.
- `misaligned = answer_correct AND iou < MISALIGN_THRESHOLD (0.3)`.

#### `POST /annotate` — `api/main.py`
- `AnnotateRequest`: all `ExplainRequest` fields + `mask` (2-D list of 0/1),
  `answer_correct`, `top_pct`.
- `AnnotateResponse`: full explain fields (overlay, grid, top regions,
  answer, confidence, latency) + `AlignmentBlock`.
- Mask validation: non-empty, rectangular, ≤ 512×512; 400 otherwise.
- `_validate_mask` helper extracts validation from the endpoint handler.

### Tests
- `tests/test_annotation.py` — 32 tests: unit (perfect/inverted/uniform
  alignment, Spearman sign, misaligned flag, resolution mismatch, error
  paths, helper unit tests) + API (happy path, block presence, value ranges,
  misaligned=False when answer_correct=False, all error contracts, lime
  method, top_pct round-trip, health regression).

### Changed
- Version bumped to `1.3.0` in `miru/__init__.py`, `miru/config.py`,
  `pyproject.toml`.
- `tests/test_api.py` health-version assertion updated to `1.3.0`.

### Test results
- **441 / 441 passing**, 5 skipped.

---

## [Unreleased] — Phase 17: cross-modal attention tracer (v1.2.0)

### Added

#### `miru/cross_modal.py` — word → image-region attribution
- `CrossModalTracer.trace(backend, image_array, question)` — for each whitespace
  token in `question`, ablates the word (removes it from the prompt), runs
  `backend.infer()` on the ablated string, and computes the positive shift in the
  spatial attention map vs. the full-question baseline.  Each row is min-max
  normalised to `[0, 1]`.
- `CrossModalTrace` frozen dataclass: `words`, `matrix` `(n_words, grid_h × grid_w)`
  float32, `grid_h`, `grid_w`, `full_attention` `(grid_h, grid_w)` float32.
- `_normalise_row` helper — min-max to `[0, 1]`; uniform → all-zero.
- Empty question → zero-row matrix without error.
- Single-word question → ablation to empty string handled (backend skipped;
  baseline treated as full attention for empty context).
- Backend-agnostic: works with any `VLMBackend`; no gradients required.

#### `POST /trace` — `api/main.py`
- `TraceRequest`: `image_b64`, `model_name` (default `"mock"`), `question`.
- `TraceResponse`: `model_name`, `question`, `words`, `matrix`
  `list[list[float]]`, `grid_h`, `grid_w`, `full_attention list[list[float]]`,
  `latency_ms`.
- Returns 400 on unknown `model_name` or undecodable `image_b64`.
- Empty question returns 200 with `words=[]` and `matrix=[]`.

### Tests
- `tests/test_cross_modal.py` — 22 tests across unit (word count, matrix
  shape/dtype/range, full-attention contracts, empty + single-word edge cases,
  determinism, inter-question variation, `_normalise_row`) and API (happy path,
  response shape, value ranges, empty question 200, unknown-model 400,
  bad-image 400, echo fields, health regression).

### Changed
- Version bumped to `1.2.0` in `miru/__init__.py`, `miru/config.py`,
  `pyproject.toml`.
- `tests/test_api.py` health-version assertion updated to `1.2.0`.

### Test results
- **409 / 409 passing**, 5 skipped (4 CLIP real-backend + 1 pre-existing).

---

## [Unreleased] — Phase 16: batch explain + content-addressed cache

### Added

#### `miru/explain_cache.py` — SQLite content-addressed cache
- `ExplainCache(path)` — thread-safe SQLite cache keyed on
  `SHA-256(image_b64 | method | model_name | params_json)`. Stores
  full ExplainResponse-shaped payloads as JSON; tracks per-row
  `hit_count` and global `total_hits` / `total_misses` counters in
  a `cache_meta` table.
- `cache_key(image_b64, method, model_name, params)` — pure helper;
  deterministic across dict orderings.
- `get_cache()` singleton honours `MIRU_CACHE_ENABLED` (default on)
  and `MIRU_CACHE_PATH` (default `./miru_cache.db`).
- Corrupt rows self-heal (deleted on first bad read).

#### `POST /explain/batch`
- Body: `{items: ExplainRequest[1..32], fidelity, record, stop_on_error}`.
- Sequential execution through the cache layer — warm items return
  instantly. Each slot is independent; one bad item doesn't fail the
  batch unless `stop_on_error=true`.
- Aggregate block: `total`, `success_count`, `failure_count`,
  `cache_hits`, `cache_misses`, `mean_confidence`, `mean_fidelity`,
  `total_latency_ms`.

#### `POST /explain` — cache-aware
- New `use_cache: bool = true` query param.
- `X-Miru-Cache: hit | miss | bypass` response header.
- Cache **hits** still call `maybe_record()` so every call produces
  its own `analysis_id` and audit-log entry — only the heavy
  compute (attention grid, overlay, top regions, fidelity block) is
  reused. Hit latency is the observed lookup time, not the cached
  compute time.

#### `GET /explain/cache_stats`
- Returns `{enabled, path, total_entries, total_hits, total_misses,
  hit_rate, size_bytes, per_method}`.

#### `POST /explain/cache_clear`
- Drops every entry, resets hit/miss counters; returns the row
  count.

### Tests
- `tests/test_explain_cache.py` — 19 unit tests covering key
  determinism + 4 partitioning axes, round-trip, hit-count column,
  corrupt-row self-heal, uncacheable-payload skip, stats, clear,
  env-gating, singleton identity + reset.
- `api/test_batch_and_cache.py` — 19 HTTP tests: cache header
  semantics (miss → hit → bypass), partition by method / param,
  env-disable, stats endpoint reflects traffic, clear endpoint,
  batch happy path / warm-cache / order / fidelity / mixed methods /
  one-bad-item-doesn't-fail-others / stop_on_error / empty 422 /
  oversized 422 / single item.

### Notes
- `/explain/compare` and `/explain/consensus` already shipped in
  Phase 13; option 2 of the brief was already done. Built options 1
  (batch) and 3 (cache) instead.
- Total tests: **387 / 387 passing**, 5 skipped. +65 net new vs the
  322 baseline before this phase.

---

## [Unreleased] — Phase 14: P1 critical sprint (fidelity, consensus, EU AI Act, export)

Audit-quality and regulator-ready output for every explanation. Four
P1 features from the researched feature roadmap, all behind the
deployable REST surface.

### Added

#### `miru/fidelity.py` — explanation-fidelity scorecard
- `deletion_test(backend, image, prompt, saliency, k_pct=0.10,
  baseline_confidence=None)` masks the top-K% salient pixels with
  per-image mean colour, re-runs `backend.infer()`, computes
  `fidelity_score = max(0, (baseline - masked) / baseline)`, clamped
  to `[0, 1]`. Optional cached baseline_confidence skips one inference.
- `FidelityResult{ fidelity_score, baseline_confidence,
  masked_confidence, k_pct, low_fidelity }`; `low_fidelity` flips on
  below `LOW_FIDELITY_THRESHOLD = 0.5`.
- Pure NumPy; inline bilinear resampler so saliency maps from any
  resolution work.

#### `miru/consensus.py` — multi-method saliency consensus
- `compute_consensus([(name, grid), ...], top_pct=0.20)` returns:
  - `agreement_grid` — `(R, R)` float in `[0, 1]`, value = fraction
    of methods that flagged each cell as top-pct
  - `consensus_score` — mean pair-wise Jaccard over top-pct masks
  - `pairwise_jaccard` — per-pair scores keyed by `"a|b"`
  - `disagreement_regions` — cells flagged by exactly one method,
    sorted descending by summed saliency
- Resamples non-matching resolutions via nearest-neighbour.

#### `miru/eu_ai_act.py` — compliance report generator
- `generate_report(record, *, system_name, provider, use_case_category)`
  maps a recorded analysis onto **Regulation (EU) 2024/1689** Articles
  11 (technical documentation), 13 (transparency), 15 (accuracy &
  robustness).
- `compliance_status` block reports per-article completeness with
  missing fields. Completeness only — human auditor sign-off still
  required, and the report says so explicitly.
- `detected_risks` flags low fidelity (< 0.5), low confidence (< 0.5),
  and method disagreement (consensus_score < 0.3).
- `COMPLIANCE_DEADLINE = "2026-08-02"` exposed as a module constant.

#### `miru/export.py` — analysis exporter
- `export_record(record, fmt)` returns `(bytes, content_type,
  suggested_filename)` for `"json"`, `"png"` (jet-colorised saliency
  at 2× via nearest-neighbour), or `"pdf"` (single-page Pillow
  document with overlay + metadata header; falls back to PNG when
  Pillow is unavailable).
- The recorder never persists source pixels, so the PNG/PDF
  deliberately renders the colorised saliency alone — never
  composited over a stored image.

#### `miru/recorder.py` — analysis_id + lookup
- `build_record(..., analysis_id=None)` auto-generates a UUID v4 when
  none is supplied; the ID is now a top-level field on every JSONL
  record.
- `maybe_record(...)` returns the recorded `analysis_id` so routes can
  echo it back to the client.
- `find_record_by_id(analysis_id, directory=None)` scans recorded
  JSONL newest-first, skips corrupt lines, returns the matching
  record or `None`.

#### `api/main.py` — four wire-format pieces
- `POST /explain?fidelity=true` — adds a `fidelity` block to the
  response (off by default — doubles the backend call count).
  `?record=true` (on by default) controls recording; the response
  now always includes `analysis_id`.
- `POST /explain/consensus` — body: image + model + `methods`
  (≥ 2, distinct, subset of IMPLEMENTED_METHODS) + per-method
  budgets. Returns per-method full results + agreement_grid +
  consensus_score + pairwise_jaccard + disagreement_regions.
- `GET /report/{analysis_id}/eu_ai_act` — returns the structured EU
  AI Act report; **404** with a hint about `MIRU_RECORD=1` on
  unknown id.
- `GET /analysis/{analysis_id}/export?format=png|json|pdf` —
  bytes + `content-type` + `Content-Disposition: attachment`.
  **400** on unknown format, **404** on unknown id.

### Tests
- `tests/test_fidelity.py` (7), `tests/test_consensus.py` (8),
  `tests/test_eu_ai_act.py` (9), `tests/test_analysis_export.py` (13),
  `api/test_api.py` (+13 new endpoint contracts).

### Test results
- **291 / 291 passing** (4 skipped CLIP real-backend tests gated on
  `MIRU_TEST_REAL_BACKENDS=1`); all 256 prior tests still green.

### Researched Feature Roadmap (recorded in PLAN.md)
- P1 critical (this sprint): fidelity scorecard, consensus overlay,
  EU AI Act report, explanation export — all shipped.
- P2 high-impact / medium-complexity: expert annotation alignment,
  dataset-level analytics, cross-modal attention tracer.
- P3 strategic: counterfactual generation, TCAV concept probes, SDK.

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
