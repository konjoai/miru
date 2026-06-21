"""Miru explainability REST API — deployable FastAPI surface.

Endpoints
---------

- ``GET  /health``                 — liveness + version + registered backends
- ``GET  /methods``                — explanation methods + backend models
- ``POST /explain``                — saliency map for one (image, model, method)
- ``POST /explain/compare``        — two methods on one image, side-by-side heatmaps
- ``POST /benchmark``              — score a backend against the synth GT-mask harness
- ``POST /compare``                — paired comparison of two methods on the same harness

Method semantics
----------------

Four explanation methods are implemented:

- ``attention`` — direct min-max-normalized extraction of the backend's
  per-patch attention weights.
- ``lime``      — superpixel-perturbation surrogate model (Ribeiro et al.
  2016), pure-NumPy implementation.  See :mod:`miru.lime_explainer`.
- ``gradcam``   — occlusion-sensitivity saliency (Zeiler & Fergus 2014),
  the gradient-free cousin of true Grad-CAM.  See
  :mod:`miru.gradcam_explainer` for the rationale on why we ship the
  black-box variant under this name.
- ``shap``      — SHAP-style tile-masking attribution (Lundberg & Lee 2017).
  Pure-NumPy sampling; no ``shap`` library required.
  See :mod:`miru.shap_explainer`.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Path as FastApiPath, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from miru.annotation import MISALIGN_THRESHOLD, compare_annotation
from miru.dataset_analytics import analyse_dataset
from miru.ensemble import AttentionEnsemble, DEFAULT_SCALES
from miru import gradcam_explainer, lime_explainer
from miru.cross_modal import CrossModalTracer
from miru.shap_explainer import SHAPConfig, SHAPExplainer
from miru.attention.extractor import AttentionExtractor
from miru.bench.comparison import compare_backends
from miru.bench.runner import run_benchmark
from miru.config import settings
from miru.consensus import compute_consensus
from miru.diff import diff_records
from miru.eu_ai_act import generate_report as generate_eu_ai_act_report
from miru.explain_cache import cache_key, get_cache, is_cache_enabled
from miru.export import SUPPORTED_FORMATS, export_record
from miru.fidelity import deletion_test
from miru.synergy import synergy_test
from miru.history import compute_calibration, query_records
from miru.model_comparison import compare_models
from miru.models import registry
from miru.posthoc_consensus import build_consensus as build_posthoc_consensus
from miru.recorder import find_record_by_id, maybe_record
from miru.search import search_by_pattern
from miru.sensitivity import DEFAULT_SIGMAS, compute_sensitivity
from miru.alerts import (
    SUPPORTED_FIELDS,
    SUPPORTED_OPS,
    fire_alerts_async,
    get_store,
    validate_webhook_url,
)
from miru.visualization.overlay import decode_image_b64, generate_overlay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits — bounded to keep a public deployment honest and predictable.
# ---------------------------------------------------------------------------

MAX_BENCH_N = 100  # cap synth-bench sample count per request
MIN_BENCH_N = 1
DEFAULT_BENCH_N = 20
DEFAULT_BENCH_SEED = 42
DEFAULT_BENCH_SIZE = 64
MAX_BENCH_SIZE = 128

IMPLEMENTED_METHODS: tuple[str, ...] = ("attention", "lime", "gradcam", "shap")
ROADMAP_METHODS: tuple[str, ...] = ()

# Bound the budgets on the perturbation-based methods so a public deploy
# can't be made to do unbounded backend.infer() calls per request.
MAX_LIME_SAMPLES = 256
MAX_LIME_SEGMENTS = 144
MAX_OCCLUSION_GRID = 16
MAX_BATCH_ITEMS = 32  # cap /explain/batch input — bounded compute, security boundary
MAX_HISTORY_LIMIT = 200  # cap /explain/history page size — bounded I/O
MAX_CALIBRATION_BINS = 50  # cap /explain/calibration bin count
MAX_COMPARE_MODELS = 8  # cap /explain/models/compare argument list — bounded I/O
MAX_POSTHOC_IDS = 16  # cap /explain/consensus/by_ids analysis_ids list
MAX_SEARCH_TOP_K = 50  # cap /explain/search top_k
MAX_SEARCH_SCAN = 2000  # cap /explain/search candidate-scan budget
MAX_SENSITIVITY_SIGMAS = 8  # cap /explain/sensitivity noise-level sweep
MAX_SENSITIVITY_TRIALS = 8  # cap perturbations per σ — bounds backend.infer() fan-out

# One extractor instance — stateless, safe across requests.
_EXTRACTOR = AttentionExtractor(resolution=settings.attention_resolution)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

registry.register_defaults()

app = FastAPI(
    title="Miru Explainability API",
    description=(
        "Saliency maps, benchmark scoring (IoU / AUC-ROC / hit@k), and "
        "method comparison for vision-language models."
    ),
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    version: str
    backends: list[str]
    methods: list[str]


class MethodInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    status: str  # "implemented" | "roadmap"
    description: str


class MethodsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    methods: list[MethodInfo]
    models: list[str]
    default_model: str


class BoundingBox(BaseModel):
    """Relative bounding box restricting saliency computation to a sub-region.

    All coordinates are fractions of the image dimensions in ``[0, 1]``.
    ``(x1, y1)`` is the top-left corner; ``(x2, y2)`` is the bottom-right.
    Both ``x2 > x1`` and ``y2 > y1`` are enforced at construction time.
    """

    model_config = ConfigDict(frozen=True)
    x1: float = Field(
        ..., ge=0.0, le=1.0, description="Left edge, fraction of image width."
    )
    y1: float = Field(
        ..., ge=0.0, le=1.0, description="Top edge, fraction of image height."
    )
    x2: float = Field(
        ..., ge=0.0, le=1.0, description="Right edge, fraction of image width."
    )
    y2: float = Field(
        ..., ge=0.0, le=1.0, description="Bottom edge, fraction of image height."
    )

    @model_validator(mode="after")
    def _check_box_order(self) -> "BoundingBox":
        """Ensure the box is non-degenerate (width > 0 and height > 0)."""
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be greater than y1 ({self.y1})")
        return self


class ExplainRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image (PNG/JPEG).")
    model_name: str = Field(
        "mock", description="Registered backend name (see /methods)."
    )
    method: str = Field(
        "attention",
        description=f"Explanation method. Implemented: {IMPLEMENTED_METHODS}.",
    )
    question: str = Field(
        "Where is the salient region?", description="Prompt to condition the backend."
    )
    alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="Heatmap opacity for the overlay."
    )
    colormap: str = Field("jet", description="One of jet | hot | viridis.")
    top_k: int = Field(
        5, ge=1, le=64, description="Number of top attention regions to return."
    )
    n_samples: int = Field(
        64, ge=2, le=MAX_LIME_SAMPLES, description="LIME perturbation count."
    )
    n_segments: int = Field(
        36, ge=4, le=MAX_LIME_SEGMENTS, description="LIME superpixel count."
    )
    occlusion_grid: int = Field(
        8, ge=2, le=MAX_OCCLUSION_GRID, description="GradCAM occlusion grid side."
    )
    shap_grid: int = Field(
        7, ge=2, le=16, description="SHAP tile grid side (shap_grid × shap_grid tiles)."
    )
    shap_samples: int = Field(
        32, ge=2, le=MAX_LIME_SAMPLES, description="SHAP coalitions sampled per tile."
    )
    roi: BoundingBox | None = Field(
        None,
        description=(
            "Optional region-of-interest bounding box (relative coords in [0, 1]). "
            "When set, saliency is computed on the cropped region and embedded back "
            "into a full-resolution grid with zeros outside the ROI. "
            "The answer/confidence always come from the full image."
        ),
    )


class TopRegion(BaseModel):
    model_config = ConfigDict(frozen=True)
    row: int
    col: int
    score: float


class FidelityBlock(BaseModel):
    """Outcome of the deletion test attached to an /explain response."""

    model_config = ConfigDict(frozen=True)
    fidelity_score: float
    baseline_confidence: float
    masked_confidence: float
    k_pct: float
    low_fidelity: bool


class SynergyBlock(BaseModel):
    """Outcome of the modality-level synergy test attached to /explain."""

    model_config = ConfigDict(frozen=True)
    synergy_score: float
    interaction: float
    f_both: float
    f_language_only: float
    f_vision_only: float
    f_neither: float
    k_pct: float
    low_synergy: bool


class ExplainResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_name: str
    method: str
    answer: str
    confidence: float
    overlay_b64: str
    attention_grid: list[list[float]]
    top_regions: list[TopRegion]
    latency_ms: float
    analysis_id: str
    fidelity: FidelityBlock | None = None
    synergy: SynergyBlock | None = None


class ConsensusRequest(BaseModel):
    """Request body for /explain/consensus.

    Runs 2-3 methods on the same image and computes their agreement.
    """

    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image.")
    model_name: str = Field("mock", description="Backend name (shared by all methods).")
    methods: list[str] = Field(
        default_factory=lambda: ["attention", "lime", "gradcam"],
        description=f"Methods to run; subset of {IMPLEMENTED_METHODS}. Minimum 2.",
    )
    question: str = Field("Where is the salient region?")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    colormap: str = Field("jet")
    top_pct: float = Field(0.20, gt=0.0, lt=1.0)
    top_k: int = Field(5, ge=1, le=64)
    n_samples: int = Field(48, ge=2, le=MAX_LIME_SAMPLES)
    n_segments: int = Field(36, ge=4, le=MAX_LIME_SEGMENTS)
    occlusion_grid: int = Field(6, ge=2, le=MAX_OCCLUSION_GRID)


class MethodResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    method: str
    overlay_b64: str
    attention_grid: list[list[float]]
    top_regions: list[TopRegion]


class ConsensusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_name: str
    methods: list[str]
    answer: str
    per_method: list[MethodResult]
    agreement_grid: list[list[float]]
    consensus_score: float
    pairwise_jaccard: dict[str, float]
    disagreement_regions: list[list[int]]
    top_pct: float
    latency_ms: float


class ExplainCompareRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image.")
    model_name: str = Field(
        "mock", description="Backend name (shared by both methods)."
    )
    method_a: str = Field("attention", description="First explanation method.")
    method_b: str = Field("gradcam", description="Second explanation method.")
    question: str = Field("Where is the salient region?")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    colormap: str = Field("jet")
    top_k: int = Field(5, ge=1, le=64)
    n_samples: int = Field(48, ge=2, le=MAX_LIME_SAMPLES)
    n_segments: int = Field(36, ge=4, le=MAX_LIME_SEGMENTS)
    shap_grid: int = Field(5, ge=2, le=16, description="SHAP tile grid side.")
    shap_samples: int = Field(
        16, ge=2, le=MAX_LIME_SAMPLES, description="SHAP coalitions per tile."
    )
    occlusion_grid: int = Field(6, ge=2, le=MAX_OCCLUSION_GRID)


class ExplainCompareResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_name: str
    method_a: str
    method_b: str
    answer: str
    a_overlay_b64: str
    b_overlay_b64: str
    a_attention_grid: list[list[float]]
    b_attention_grid: list[list[float]]
    a_top_regions: list[TopRegion]
    b_top_regions: list[TopRegion]
    latency_ms: float


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_name: str = Field("mock", description="Backend to score.")
    n: int = Field(DEFAULT_BENCH_N, ge=MIN_BENCH_N, le=MAX_BENCH_N)
    seed: int = Field(DEFAULT_BENCH_SEED, description="Synth-dataset seed.")
    size: int = Field(DEFAULT_BENCH_SIZE, ge=16, le=MAX_BENCH_SIZE)
    top_pct: float = Field(0.20, gt=0.0, lt=1.0)
    k_for_hit: int = Field(1, ge=1, le=64)


class MetricStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    mean: float
    std: float
    p50: float
    p95: float
    n: int


class BenchmarkResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: str
    n: int
    seed: int
    size: int
    iou: MetricStats
    auc: MetricStats
    hit1: MetricStats
    latency_ms: MetricStats
    timestamp_utc: str


class CompareRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_a: str = Field(..., description="First backend name.")
    model_b: str = Field(..., description="Second backend name.")
    n: int = Field(DEFAULT_BENCH_N, ge=MIN_BENCH_N, le=MAX_BENCH_N)
    seed: int = Field(DEFAULT_BENCH_SEED)


class CompareResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_a: str
    model_b: str
    winner: str
    n: int
    seed: int
    a_iou: MetricStats
    b_iou: MetricStats
    a_auc: MetricStats
    b_auc: MetricStats
    a_hit1: MetricStats
    b_hit1: MetricStats
    paired_iou_delta: float
    paired_iou_t_statistic: float
    timestamp: str


# ----- Batch + cache schemas -------------------------------------------------


class BatchExplainRequest(BaseModel):
    """Body for ``POST /explain/batch``.

    Each item is a full :class:`ExplainRequest` so callers can mix
    methods, models, prompts, and per-item knobs within one batch.
    The ``fidelity`` / ``record`` flags apply uniformly to every item
    (matching how clients use the single-image endpoint).
    """

    model_config = ConfigDict(frozen=True)
    items: list[ExplainRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_ITEMS,
        description=f"Per-image requests. 1..{MAX_BATCH_ITEMS}.",
    )
    fidelity: bool = Field(
        False,
        description="Run the deletion test on every item.",
    )
    synergy: bool = Field(
        False,
        description="Run the modality-level synergy test on every item.",
    )
    record: bool = Field(
        True,
        description="Persist a JSONL trace per item (no-op when MIRU_RECORD unset).",
    )
    stop_on_error: bool = Field(
        False,
        description=(
            "When true, the batch aborts on the first item that fails. "
            "Following items are returned with ``success=False`` and a "
            "skipped error."
        ),
    )


class BatchItemResult(BaseModel):
    """One slot in :class:`BatchExplainResponse.items`."""

    model_config = ConfigDict(frozen=True)
    index: int
    success: bool
    cached: bool
    response: ExplainResponse | None = None
    error: str | None = None


class BatchAggregate(BaseModel):
    """Roll-up statistics for one batch run."""

    model_config = ConfigDict(frozen=True)
    total: int
    success_count: int
    failure_count: int
    cache_hits: int
    cache_misses: int
    mean_confidence: float | None
    mean_fidelity: float | None
    total_latency_ms: float


class BatchExplainResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    items: list[BatchItemResult]
    aggregate: BatchAggregate


class CacheStatsResponse(BaseModel):
    """Body of ``GET /explain/cache_stats``."""

    model_config = ConfigDict(frozen=True)
    enabled: bool
    path: str | None = None
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float | None = None
    size_bytes: int = 0
    per_method: dict[str, int] = Field(default_factory=dict)


class CacheClearResponse(BaseModel):
    """Body of ``POST /explain/cache_clear``."""

    model_config = ConfigDict(frozen=True)
    cleared: int


# ----- History + calibration + diff schemas ----------------------------------


class HistoryItem(BaseModel):
    """One row in :class:`HistoryResponse.items`."""

    model_config = ConfigDict(frozen=True)
    analysis_id: str
    ts: str
    question: str
    image_sha256: str | None
    backend: str
    method: str
    confidence: float
    latency_ms: float
    fidelity_score: float | None
    cache_hit: bool


class HistoryResponse(BaseModel):
    """Paginated /explain/history payload."""

    model_config = ConfigDict(frozen=True)
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


class CalibrationBinModel(BaseModel):
    """One bucket on the calibration curve."""

    model_config = ConfigDict(frozen=True)
    lo: float
    hi: float
    count: int
    mean_confidence: float
    mean_fidelity: float
    gap: float


class CalibrationResponse(BaseModel):
    """Body of ``GET /explain/calibration``."""

    model_config = ConfigDict(frozen=True)
    n: int
    n_bins: int
    ece: float
    mean_confidence: float
    mean_fidelity: float
    bins: list[CalibrationBinModel]
    # Optional filter echo so the client can confirm what got aggregated.
    filter_method: str | None = None
    filter_model: str | None = None


class DiffRequest(BaseModel):
    """Body of ``POST /explain/diff``."""

    model_config = ConfigDict(frozen=True)
    analysis_id_a: str = Field(..., min_length=8, max_length=64)
    analysis_id_b: str = Field(..., min_length=8, max_length=64)
    top_n: int = Field(10, ge=1, le=64, description="Top changed cells to return.")


class TopChangedRegionModel(BaseModel):
    """One cell where attribution shifted most."""

    model_config = ConfigDict(frozen=True)
    row: int
    col: int
    value_a: float
    value_b: float
    delta: float


class DiffResponse(BaseModel):
    """Body of ``POST /explain/diff``."""

    model_config = ConfigDict(frozen=True)
    analysis_id_a: str
    analysis_id_b: str
    method_a: str
    method_b: str
    backend_a: str
    backend_b: str
    cosine_similarity: float
    l2_distance: float
    delta_grid: list[list[float]]
    top_changed: list[TopChangedRegionModel]
    summary: str


# ----- Model-comparison schemas ----------------------------------------------


class ModelStatsBlock(BaseModel):
    """One model's row in :class:`ModelsCompareResponse.stats`."""

    model_config = ConfigDict(frozen=True)
    model: str
    n_records: int
    mean_confidence: float | None
    mean_latency_ms: float | None
    mean_fidelity: float | None
    n_with_fidelity: int
    ece: float | None
    method_distribution: dict[str, int]


class ModelsCompareResponse(BaseModel):
    """Body of ``GET /explain/models/compare``."""

    model_config = ConfigDict(frozen=True)
    models: list[str]
    stats: dict[str, ModelStatsBlock]
    winner_by_confidence: str | None
    winner_by_fidelity: str | None
    winner_by_ece: str | None
    filter_method: str | None = None
    limit: int


# ----- Post-hoc consensus schemas --------------------------------------------


class PosthocConsensusRequest(BaseModel):
    """Body of ``POST /explain/consensus/by_ids``."""

    model_config = ConfigDict(frozen=True)
    analysis_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=MAX_POSTHOC_IDS,
        description=f"Distinct analysis_ids to combine (2..{MAX_POSTHOC_IDS}).",
    )
    weighting: str = Field(
        "fidelity",
        description="One of: fidelity | confidence | uniform.",
    )
    top_k: int = Field(5, ge=1, le=64, description="Top consensus cells to return.")


class PerRecordContributionModel(BaseModel):
    """One slot in :class:`PosthocConsensusResponse.per_record`."""

    model_config = ConfigDict(frozen=True)
    analysis_id: str
    method: str
    backend: str
    weight: float
    agreement_score: float


class TopConsensusRegionModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    row: int
    col: int
    score: float


class PosthocConsensusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    consensus_grid: list[list[float]]
    per_record: list[PerRecordContributionModel]
    top_regions: list[TopConsensusRegionModel]
    weighting: str
    n_records: int
    grid_h: int
    grid_w: int


# ----- Search-by-pattern schemas --------------------------------------------


class SearchRequest(BaseModel):
    """Body of ``POST /explain/search``.

    Exactly one of ``query_grid`` and ``query_analysis_id`` must be set.
    """

    model_config = ConfigDict(frozen=True)
    query_grid: list[list[float]] | None = Field(
        None,
        description="2-D float saliency map used as the query.",
    )
    query_analysis_id: str | None = Field(
        None,
        description="Pull the query grid from this recorded analysis.",
    )
    method: str | None = Field(
        None,
        description="Restrict candidates to this explanation method.",
    )
    model: str | None = Field(
        None,
        description="Restrict candidates to this backend.",
    )
    top_k: int = Field(10, ge=1, le=MAX_SEARCH_TOP_K)
    min_similarity: float | None = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Drop matches with cosine below this value.",
    )
    max_scan: int = Field(
        500,
        ge=1,
        le=MAX_SEARCH_SCAN,
        description="Maximum candidates to score before slicing.",
    )


class SearchMatchModel(BaseModel):
    """One match row."""

    model_config = ConfigDict(frozen=True)
    analysis_id: str
    ts: str
    method: str
    backend: str
    question: str
    similarity: float


class SearchResponse(BaseModel):
    """Body of ``POST /explain/search``."""

    model_config = ConfigDict(frozen=True)
    matches: list[SearchMatchModel]
    n_candidates: int
    n_scanned: int
    top_k: int
    query_analysis_id: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        backends=registry.available(),
        methods=list(IMPLEMENTED_METHODS),
    )


_METHOD_DESCRIPTIONS: dict[str, str] = {
    "attention": (
        "Direct min-max-normalized attention-weight extraction from the "
        "VLM backend, resampled to a fixed grid."
    ),
    "lime": (
        "LIME (Ribeiro et al. 2016): superpixel perturbation + weighted "
        "least-squares surrogate model. Pure-NumPy implementation."
    ),
    "gradcam": (
        "Occlusion-sensitivity saliency (Zeiler & Fergus 2014): grid-occlude "
        "the image and measure attention shift per cell. The gradient-free "
        "cousin of true Grad-CAM; backend-agnostic."
    ),
    "shap": (
        "SHAP-style tile-masking attribution (Lundberg & Lee 2017): "
        "estimates φᵢ ≈ E[f(x)|xᵢ present] − E[f(x)|xᵢ absent] by "
        "sampling random tile coalitions.  Pure-NumPy, no shap library."
    ),
}


@app.get("/methods", response_model=MethodsResponse)
def methods() -> MethodsResponse:
    info: list[MethodInfo] = [
        MethodInfo(
            name=m,
            status="implemented",
            description=_METHOD_DESCRIPTIONS.get(m, ""),
        )
        for m in IMPLEMENTED_METHODS
    ] + [
        MethodInfo(
            name=m,
            status="roadmap",
            description=_METHOD_DESCRIPTIONS.get(
                m, f"{m} is on the roadmap; not yet implemented in miru."
            ),
        )
        for m in ROADMAP_METHODS
    ]
    return MethodsResponse(
        methods=info,
        models=registry.available(),
        default_model=settings.default_backend,
    )


def _explain_cache_key(req: ExplainRequest, fidelity: bool, synergy: bool) -> str:
    """Cache key covering every input that materially affects the response."""
    roi_key = (
        {"x1": req.roi.x1, "y1": req.roi.y1, "x2": req.roi.x2, "y2": req.roi.y2}
        if req.roi is not None
        else None
    )
    params = {
        "question": req.question,
        "alpha": req.alpha,
        "colormap": req.colormap,
        "top_k": req.top_k,
        "n_samples": req.n_samples,
        "n_segments": req.n_segments,
        "occlusion_grid": req.occlusion_grid,
        "shap_grid": req.shap_grid,
        "shap_samples": req.shap_samples,
        "fidelity": bool(fidelity),
        "synergy": bool(synergy),
        "roi": roi_key,
    }
    return cache_key(req.image_b64, req.method, req.model_name, params)


def _apply_roi_saliency(
    full_image: np.ndarray,
    roi: BoundingBox,
    method: str,
    backend: object,
    req: ExplainRequest,
) -> tuple[object, np.ndarray]:
    """Run the chosen explainer on the ROI crop; embed result in a full-size grid.

    The VLM answer always comes from the *full* image so it reflects the whole
    scene.  Saliency is computed only on the cropped pixels, confining attribution
    to the area of interest.  Non-ROI cells in the returned grid are zero.

    Raises ``HTTPException(400)`` when the crop is smaller than 4×4 pixels.
    """
    resolution = settings.attention_resolution
    H, W = full_image.shape[:2]

    px1 = int(roi.x1 * W)
    py1 = int(roi.y1 * H)
    px2 = max(px1 + 1, min(W, int(roi.x2 * W)))
    py2 = max(py1 + 1, min(H, int(roi.y2 * H)))

    crop_h, crop_w = py2 - py1, px2 - px1
    if crop_h < 4 or crop_w < 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"roi maps to a {crop_h}×{crop_w} pixel crop on this image "
                "(min 4×4 required). Expand the bounding box."
            ),
        )

    crop = full_image[py1:py2, px1:px2]

    full_out = backend.infer(full_image, req.question)  # type: ignore[union-attr]

    _, saliency_crop = _run_method(method, backend, crop, req)

    gc1 = int(roi.x1 * resolution)
    gr1 = int(roi.y1 * resolution)
    gc2 = max(gc1 + 1, min(resolution, int(roi.x2 * resolution)))
    gr2 = max(gr1 + 1, min(resolution, int(roi.y2 * resolution)))

    scaled = _EXTRACTOR.resize_to_grid(saliency_crop, gr2 - gr1, gc2 - gc1)

    full_grid = np.zeros((resolution, resolution), dtype=np.float32)
    full_grid[gr1:gr2, gc1:gc2] = scaled
    return full_out, full_grid


def _evaluate_and_fire_alerts(
    analysis_id: str,
    confidence: float,
    fidelity_dict: dict[str, object] | None,
) -> None:
    """Evaluate alert rules against this analysis result and fire webhooks."""
    store = get_store()
    if store is None:
        return
    result: dict[str, object] = {"confidence": confidence}
    if fidelity_dict is not None:
        result["fidelity"] = fidelity_dict
    try:
        fired = store.evaluate(analysis_id, result)
    except (ValueError, OSError) as exc:
        logger.warning("alert evaluation failed for %s: %s", analysis_id, exc)
        return
    if fired:
        fire_alerts_async(fired, store)


def _run_explain_uncached(
    req: ExplainRequest,
    *,
    fidelity: bool,
    synergy: bool,
    record: bool,
) -> dict[str, object]:
    """Run one /explain end-to-end and return a dict ready for ExplainResponse.

    Pure compute path — no cache lookup, no cache write. The caller wraps
    this when caching is enabled.
    """
    _validate_method(req.method)
    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    t0 = time.perf_counter()
    if req.roi is not None:
        out, saliency_grid = _apply_roi_saliency(
            image_array, req.roi, req.method, backend, req
        )
    else:
        out, saliency_grid = _run_method(req.method, backend, image_array, req)
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    top = _EXTRACTOR.top_k_regions(saliency_grid, k=req.top_k)

    try:
        overlay_b64 = generate_overlay(
            req.image_b64, saliency_grid, alpha=req.alpha, colormap=req.colormap
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"overlay generation failed: {exc}"
        ) from exc

    fidelity_dict: dict[str, object] | None = None
    if fidelity:
        result = deletion_test(
            backend,
            image_array,
            req.question,
            saliency_grid,
            baseline_confidence=float(out.confidence),
        )
        fidelity_dict = {
            "fidelity_score": result.fidelity_score,
            "baseline_confidence": result.baseline_confidence,
            "masked_confidence": result.masked_confidence,
            "k_pct": result.k_pct,
            "low_fidelity": result.low_fidelity,
        }

    synergy_dict: dict[str, object] | None = None
    if synergy:
        syn = synergy_test(
            backend,
            image_array,
            req.question,
            saliency_grid,
            baseline_confidence=float(out.confidence),
        )
        synergy_dict = {
            "synergy_score": syn.synergy_score,
            "interaction": syn.interaction,
            "f_both": syn.f_both,
            "f_language_only": syn.f_language_only,
            "f_vision_only": syn.f_vision_only,
            "f_neither": syn.f_neither,
            "k_pct": syn.k_pct,
            "low_synergy": syn.low_synergy,
        }

    attention_grid_list = saliency_grid.astype(float).tolist()
    top_list = [{"row": int(r), "col": int(c), "score": float(s)} for r, c, s in top]

    # Build a stable trace dict for the recorder.
    trace_dict = {
        "answer": out.answer,
        "confidence": float(out.confidence),
        "backend": backend.name,
        "method": req.method,
        "explanation_method": req.method,
        "latency_ms": latency_ms,
        "attention_grid": attention_grid_list,
        "top_regions": top_list,
        "fidelity": fidelity_dict,
        "synergy": synergy_dict,
    }
    analysis_id = (
        maybe_record(trace_dict, image_b64=req.image_b64, question=req.question)
        if record
        else None
    ) or ""

    # Evaluate alert rules (non-blocking; webhook delivery is async).
    _evaluate_and_fire_alerts(analysis_id, float(out.confidence), fidelity_dict)

    return {
        "model_name": backend.name,
        "method": req.method,
        "answer": out.answer,
        "confidence": float(out.confidence),
        "overlay_b64": overlay_b64,
        "attention_grid": attention_grid_list,
        "top_regions": top_list,
        "latency_ms": latency_ms,
        "analysis_id": analysis_id,
        "fidelity": fidelity_dict,
        "synergy": synergy_dict,
    }


def _run_explain_with_cache(
    req: ExplainRequest,
    *,
    fidelity: bool,
    synergy: bool,
    record: bool,
) -> tuple[ExplainResponse, bool]:
    """Run /explain through the cache. Returns (response, was_cache_hit).

    Cache semantics
    ---------------
    The cache stores the heavy computation — saliency grid, overlay,
    top regions, answer, confidence, fidelity block — keyed on the
    inputs that materially affect those fields.

    Two things are *not* preserved from the cached payload:

    1. ``analysis_id``: each call to ``/explain`` is a distinct audit
       event and must produce its own ID. On cache hit we still call
       ``maybe_record`` so the audit log records this specific call,
       then substitute the fresh ID into the returned response.
    2. ``latency_ms``: the cached value reflects the *original* call's
       compute time. Cache hits return a near-zero latency to make the
       observable speedup honest to clients.
    """
    cache = get_cache()
    if cache is None:
        return ExplainResponse(
            **_run_explain_uncached(
                req, fidelity=fidelity, synergy=synergy, record=record
            )
        ), False

    key = _explain_cache_key(req, fidelity, synergy)
    t0 = time.perf_counter()
    cached = cache.get(key)
    if cached is not None:
        # Re-record so every call is audited, even cache hits.
        trace_dict = {
            "answer": cached["answer"],
            "confidence": cached["confidence"],
            "backend": cached["model_name"],
            "method": cached["method"],
            "explanation_method": cached["method"],
            "latency_ms": cached.get("latency_ms", 0.0),
            "attention_grid": cached["attention_grid"],
            "top_regions": cached["top_regions"],
            "fidelity": cached.get("fidelity"),
            "synergy": cached.get("synergy"),
            "cache_hit": True,
        }
        fresh_id = (
            maybe_record(trace_dict, image_b64=req.image_b64, question=req.question)
            if record
            else None
        ) or ""
        # Build a response with the fresh ID + observed lookup latency.
        merged = {
            **cached,
            "analysis_id": fresh_id,
            "latency_ms": (time.perf_counter() - t0) * 1_000.0,
        }
        _evaluate_and_fire_alerts(
            fresh_id,
            float(cached["confidence"]),
            cached.get("fidelity"),  # type: ignore[arg-type]
        )
        return ExplainResponse(**merged), True

    payload = _run_explain_uncached(
        req, fidelity=fidelity, synergy=synergy, record=record
    )
    cache.put(key, payload, method=req.method, model_name=req.model_name)
    return ExplainResponse(**payload), False


@app.post("/explain", response_model=ExplainResponse)
def explain(
    req: ExplainRequest,
    response: Response,
    fidelity: bool = Query(
        default=False,
        description=(
            "Run the deletion test on the saliency map and include a "
            "fidelity block in the response. Doubles the backend call "
            "count, so off by default."
        ),
    ),
    synergy: bool = Query(
        default=False,
        description=(
            "Run the modality-level synergy test (vision×language Shapley "
            "interaction) and include a synergy block in the response. "
            "Adds three extra backend calls, so off by default. A low "
            "synergy_score flags visual-only salience rather than faithful "
            "cross-modal reasoning."
        ),
    ),
    record: bool = Query(
        default=True,
        description=(
            "Persist a privacy-stripped JSONL trace via the recorder "
            "(no-op when MIRU_RECORD is unset). The returned analysis_id "
            "can later be passed to /report/{id}/eu_ai_act and "
            "/analysis/{id}/export."
        ),
    ),
    use_cache: bool = Query(
        default=True,
        description=(
            "When true (default), serve cached results for repeat "
            "(image, model, method, params) combinations. Set false "
            "to force re-computation."
        ),
    ),
) -> ExplainResponse:
    """Run one explanation. Cache-aware unless ``use_cache=false`` is passed.

    The response header ``X-Miru-Cache`` is set to ``hit`` or ``miss`` so
    clients can observe cache behaviour without parsing the JSON.
    """
    if use_cache and is_cache_enabled():
        result, was_hit = _run_explain_with_cache(
            req, fidelity=fidelity, synergy=synergy, record=record
        )
        response.headers["X-Miru-Cache"] = "hit" if was_hit else "miss"
        return result
    response.headers["X-Miru-Cache"] = "bypass"
    return ExplainResponse(
        **_run_explain_uncached(
            req, fidelity=fidelity, synergy=synergy, record=record
        )
    )


@app.post("/explain/compare", response_model=ExplainCompareResponse)
def explain_compare(req: ExplainCompareRequest) -> ExplainCompareResponse:
    """Run two methods on the same image and return both saliency overlays.

    Used by the demo UI for side-by-side comparison (e.g. attention vs lime).
    Both methods see the same image and the same backend so the only
    moving part across the two heatmaps is the explanation method itself.
    """
    if req.method_a == req.method_b:
        raise HTTPException(
            status_code=400,
            detail=f"method_a and method_b must differ; both were '{req.method_a}'.",
        )
    _validate_method(req.method_a)
    _validate_method(req.method_b)
    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    t0 = time.perf_counter()
    out_a, sal_a = _run_method(req.method_a, backend, image_array, req)
    out_b, sal_b = _run_method(req.method_b, backend, image_array, req)
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    try:
        overlay_a = generate_overlay(
            req.image_b64, sal_a, alpha=req.alpha, colormap=req.colormap
        )
        overlay_b = generate_overlay(
            req.image_b64, sal_b, alpha=req.alpha, colormap=req.colormap
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"overlay generation failed: {exc}"
        ) from exc

    top_a = _EXTRACTOR.top_k_regions(sal_a, k=req.top_k)
    top_b = _EXTRACTOR.top_k_regions(sal_b, k=req.top_k)

    return ExplainCompareResponse(
        model_name=backend.name,
        method_a=req.method_a,
        method_b=req.method_b,
        answer=out_a.answer,
        a_overlay_b64=overlay_a,
        b_overlay_b64=overlay_b,
        a_attention_grid=sal_a.astype(float).tolist(),
        b_attention_grid=sal_b.astype(float).tolist(),
        a_top_regions=[TopRegion(row=r, col=c, score=s) for r, c, s in top_a],
        b_top_regions=[TopRegion(row=r, col=c, score=s) for r, c, s in top_b],
        latency_ms=latency_ms,
    )


@app.post("/explain/consensus", response_model=ConsensusResponse)
def explain_consensus(req: ConsensusRequest) -> ConsensusResponse:
    """Run 2-3 methods on one image and compute their saliency consensus.

    Returns each method's full overlay + grid + top regions, plus an
    `agreement_grid` whose value is the fraction of methods that
    flagged each cell as top-`top_pct`, the mean pair-wise Jaccard
    consensus score, and the explicit list of `disagreement_regions`
    flagged in exactly one method.
    """
    if len(req.methods) < 2:
        raise HTTPException(
            status_code=400,
            detail="consensus requires at least two methods",
        )
    if len(set(req.methods)) != len(req.methods):
        raise HTTPException(
            status_code=400,
            detail="methods must be distinct",
        )
    for m in req.methods:
        _validate_method(m)

    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    # Translate the consensus request shape into the per-method arg
    # bundle that _run_method expects.
    inner_req = ExplainRequest(
        image_b64=req.image_b64,
        model_name=req.model_name,
        method=req.methods[0],  # placeholder; overwritten per call
        question=req.question,
        alpha=req.alpha,
        colormap=req.colormap,
        top_k=req.top_k,
        n_samples=req.n_samples,
        n_segments=req.n_segments,
        occlusion_grid=req.occlusion_grid,
    )

    t0 = time.perf_counter()
    per_method: list[tuple[str, np.ndarray, MethodResult]] = []
    answer = ""
    for method in req.methods:
        out, sal = _run_method(method, backend, image_array, inner_req)
        if not answer:
            answer = out.answer
        try:
            overlay = generate_overlay(
                req.image_b64, sal, alpha=req.alpha, colormap=req.colormap
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"overlay generation failed: {exc}"
            ) from exc
        top = _EXTRACTOR.top_k_regions(sal, k=req.top_k)
        per_method.append(
            (
                method,
                sal,
                MethodResult(
                    method=method,
                    overlay_b64=overlay,
                    attention_grid=sal.astype(float).tolist(),
                    top_regions=[TopRegion(row=r, col=c, score=s) for r, c, s in top],
                ),
            )
        )

    consensus = compute_consensus(
        [(name, sal) for name, sal, _ in per_method],
        top_pct=req.top_pct,
    )
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    return ConsensusResponse(
        model_name=backend.name,
        methods=req.methods,
        answer=answer,
        per_method=[m for _, _, m in per_method],
        agreement_grid=consensus.agreement_grid.astype(float).tolist(),
        consensus_score=consensus.consensus_score,
        pairwise_jaccard=consensus.pairwise_jaccard,
        disagreement_regions=[[r, c] for r, c in consensus.disagreement_regions],
        top_pct=consensus.top_pct,
        latency_ms=latency_ms,
    )


@app.get("/report/{analysis_id}/eu_ai_act")
def eu_ai_act_report(
    analysis_id: str = FastApiPath(
        ..., min_length=8, description="analysis_id returned by /explain"
    ),
) -> dict:
    """EU AI Act compliance report for one recorded analysis.

    Looks the analysis up by ID in the recorder's JSONL store; returns
    a structured report covering Article 11 (technical documentation),
    Article 12 (record-keeping), Article 13 (transparency, incl. documented
    feature importance), Article 15 (accuracy & robustness, incl. the
    cross-modal synergy probe), and Article 86 (right to explanation).

    Returns 404 when the analysis_id is not present in the record store
    (which most often means recording was disabled at /explain time).
    """
    record = find_record_by_id(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"analysis_id '{analysis_id}' not found; "
                "ensure MIRU_RECORD=1 was set when the analysis ran"
            ),
        )
    return generate_eu_ai_act_report(record)


@app.get("/analysis/{analysis_id}/export")
def analysis_export(
    analysis_id: str = FastApiPath(
        ..., min_length=8, description="analysis_id returned by /explain"
    ),
    format: str = Query(  # noqa: A002 — matches the public API name in the spec
        default="json",
        description=f"Export format; one of {SUPPORTED_FORMATS}.",
    ),
) -> Response:
    """Export a recorded analysis as PNG, JSON, or PDF.

    PNG: heatmap colorised at 2× via the jet palette.
    JSON: the full recorded JSONL record.
    PDF: single-page Pillow document with the overlay and a metadata
    header (falls back to PNG bytes when Pillow is unavailable).
    """
    if format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(f"format must be one of {SUPPORTED_FORMATS}, got {format!r}"),
        )
    record = find_record_by_id(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"analysis_id '{analysis_id}' not found; "
                "ensure MIRU_RECORD=1 was set when the analysis ran"
            ),
        )
    payload, content_type, filename = export_record(record, format)
    return Response(
        content=payload,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    if req.model_name not in registry.available():
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown model_name='{req.model_name}'. "
                f"Available: {registry.available()}."
            ),
        )

    result = run_benchmark(
        backend_name=req.model_name,
        n=req.n,
        seed=req.seed,
        size=req.size,
        top_pct=req.top_pct,
        k_for_hit=req.k_for_hit,
    )
    agg = result["aggregate"]
    return BenchmarkResponse(
        backend=result["backend"],
        n=result["n"],
        seed=result["seed"],
        size=result["size"],
        iou=MetricStats(**agg["iou"]),
        auc=MetricStats(**agg["auc"]),
        hit1=MetricStats(**agg["hit1"]),
        latency_ms=MetricStats(**agg["latency_ms"]),
        timestamp_utc=result["timestamp_utc"],
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    avail = registry.available()
    for name in (req.model_a, req.model_b):
        if name not in avail:
            raise HTTPException(
                status_code=400,
                detail=f"unknown model='{name}'. Available: {avail}.",
            )

    bc = compare_backends(
        backend_a_name=req.model_a,
        backend_b_name=req.model_b,
        n_samples=req.n,
        seed=req.seed,
        save=False,
    )

    agg_a = bc.result_a["aggregate"]
    agg_b = bc.result_b["aggregate"]
    cmp = bc.comparison or {}
    return CompareResponse(
        model_a=req.model_a,
        model_b=req.model_b,
        winner=bc.winner,
        n=req.n,
        seed=req.seed,
        a_iou=MetricStats(**agg_a["iou"]),
        b_iou=MetricStats(**agg_b["iou"]),
        a_auc=MetricStats(**agg_a["auc"]),
        b_auc=MetricStats(**agg_b["auc"]),
        a_hit1=MetricStats(**agg_a["hit1"]),
        b_hit1=MetricStats(**agg_b["hit1"]),
        paired_iou_delta=float(cmp.get("mean_delta", 0.0)),
        paired_iou_t_statistic=float(cmp.get("t_statistic", 0.0)),
        timestamp=bc.timestamp,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_method(method: str) -> None:
    if method in IMPLEMENTED_METHODS:
        return
    if method in ROADMAP_METHODS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"method='{method}' is on the roadmap but not yet implemented. "
                f"Use one of: {list(IMPLEMENTED_METHODS)}."
            ),
        )
    raise HTTPException(
        status_code=400,
        detail=f"unknown method='{method}'. Use one of: {list(IMPLEMENTED_METHODS)}.",
    )


def _get_backend_or_400(model_name: str):
    try:
        return registry.get(model_name)
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown model_name='{model_name}'. Available: {registry.available()}."
            ),
        ) from exc


def _run_method(method: str, backend, image_array: np.ndarray, req):
    """Dispatch to the chosen explainer.

    Returns ``(VLMOutput-like, saliency_grid)``.  All methods produce a
    ``(resolution, resolution)`` float32 saliency in ``[0, 1]``.  For
    LIME and GradCAM we still return the backend's text answer + confidence
    from the baseline ``infer()`` call so the API contract stays uniform.
    """
    if method == "attention":
        out = backend.infer(image_array, req.question)
        return out, _EXTRACTOR.extract(out.attention_weights)

    if method == "lime":
        baseline = backend.infer(image_array, req.question)
        result = lime_explainer.explain(
            backend,
            image_array,
            req.question,
            n_segments=req.n_segments,
            n_samples=req.n_samples,
            resolution=settings.attention_resolution,
        )
        return baseline, result.saliency

    if method == "gradcam":
        baseline = backend.infer(image_array, req.question)
        result = gradcam_explainer.explain(
            backend,
            image_array,
            req.question,
            occlusion_grid=req.occlusion_grid,
            resolution=settings.attention_resolution,
        )
        return baseline, result.saliency

    if method == "shap":
        baseline = backend.infer(image_array, req.question)
        cfg = SHAPConfig(
            grid_size=req.shap_grid,
            n_samples=req.shap_samples,
            seed=42,
        )
        pil_image = _float_array_to_pil(image_array)
        attribution = SHAPExplainer(backend, cfg).explain(pil_image, req.question)
        # Shift [-1, 1] → [0, 1] for the overlay pipeline.
        norm = (attribution.astype(np.float32) + 1.0) / 2.0
        saliency = _EXTRACTOR.resize_to_grid(
            norm, settings.attention_resolution, settings.attention_resolution
        )
        return baseline, saliency

    raise HTTPException(status_code=400, detail=f"unsupported method: {method}")


def _validate_mask(mask: list[list[float]]) -> np.ndarray:
    """Validate and convert a 2-D mask list to a bool numpy array.

    Raises HTTP 400 on empty mask, non-rectangular rows, or oversized dims.
    """
    if not mask:
        raise HTTPException(status_code=400, detail="mask must not be empty")
    n_rows = len(mask)
    n_cols = len(mask[0]) if mask else 0
    if n_rows > MAX_MASK_DIM or n_cols > MAX_MASK_DIM:
        raise HTTPException(
            status_code=400,
            detail=(
                f"mask dimensions {n_rows}×{n_cols} exceed the limit "
                f"{MAX_MASK_DIM}×{MAX_MASK_DIM}."
            ),
        )
    for i, row in enumerate(mask):
        if len(row) != n_cols:
            raise HTTPException(
                status_code=400,
                detail=f"mask row {i} has {len(row)} columns; expected {n_cols}.",
            )
    return np.array(mask, dtype=np.float32)


def _decode_to_float_array(image_b64: str) -> np.ndarray:
    """Decode a base64 image to a float32 (H, W, 3) array in [0, 1].

    Raises HTTP 400 on any decode failure — this API surface accepts
    real image formats (PNG/JPEG/etc.) only, so malformed input is a
    client error worth reporting clearly.
    """
    try:
        rgba = decode_image_b64(image_b64)
    except Exception as exc:  # noqa: BLE001 — boundary code
        logger.info("image decode failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"image_b64 is not a decodable image: {exc}",
        ) from exc
    return rgba[..., :3].astype(np.float32) / 255.0


def _float_array_to_pil(image_array: np.ndarray) -> "Image.Image":  # noqa: F821
    """Convert a float32 (H, W, 3) ∈ [0, 1] array to a PIL RGB Image."""
    from PIL import Image

    uint8 = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


# ---------------------------------------------------------------------------
# Scale-space attention ensemble endpoint
# ---------------------------------------------------------------------------

MAX_ENSEMBLE_SCALES = 5


class EnsembleRequest(BaseModel):
    """Request body for ``POST /explain/ensemble``."""

    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image (PNG/JPEG).")
    model_name: str = Field("mock", description="Registered backend name.")
    question: str = Field("Where is the salient region?")
    scales: list[float] = Field(
        default_factory=lambda: list(DEFAULT_SCALES),
        min_length=1,
        max_length=MAX_ENSEMBLE_SCALES,
        description=(
            f"Scale factors relative to the input image. 1..{MAX_ENSEMBLE_SCALES} values, "
            "each in (0, 4]."
        ),
    )
    weights: list[float] | None = Field(
        None,
        description=(
            "Optional per-scale weights (same length as scales). "
            "Defaults to uniform weighting."
        ),
    )
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    colormap: str = Field("jet")
    top_k: int = Field(5, ge=1, le=64)


class PerScaleResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    scale: float
    attention_grid: list[list[float]]


class EnsembleResponse(BaseModel):
    """Response body for ``POST /explain/ensemble``."""

    model_config = ConfigDict(frozen=True)
    model_name: str
    question: str
    scales_requested: list[float]
    scales_used: list[float]
    scales_skipped: list[float]
    ensemble_grid: list[list[float]]
    per_scale: list[PerScaleResult]
    top_regions: list[TopRegion]
    overlay_b64: str
    latency_ms: float


@app.post("/explain/ensemble", response_model=EnsembleResponse)
def explain_ensemble(req: EnsembleRequest) -> EnsembleResponse:
    """Multi-scale attention ensemble for more robust saliency maps.

    Runs the backend at each requested scale factor (relative to the input
    image size), normalises each attention map to a fixed grid, and produces
    a weighted average.  The result is more robust than single-scale attention
    because models are sensitive to input resolution — aggregating across
    scales captures a larger fraction of the true saliency signal.

    Scales that produce images below 4 pixels in either dimension are silently
    skipped.  The ``scales_skipped`` field in the response reports which ones
    were dropped.
    """
    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    if req.weights is not None and len(req.weights) != len(req.scales):
        raise HTTPException(
            status_code=400,
            detail=(
                f"weights length ({len(req.weights)}) must match "
                f"scales length ({len(req.scales)})."
            ),
        )
    for s in req.scales:
        if not (0.0 < s <= 4.0):
            raise HTTPException(
                status_code=400,
                detail=f"each scale must be in (0, 4], got {s!r}.",
            )

    weights_tuple = tuple(req.weights) if req.weights is not None else None
    ensembler = AttentionEnsemble(
        scales=tuple(req.scales),
        weights=weights_tuple,
    )

    t0 = time.perf_counter()
    result = ensembler.run(backend, image_array, req.question)
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    top = _EXTRACTOR.top_k_regions(result.ensemble_grid, k=req.top_k)
    try:
        overlay_b64 = generate_overlay(
            req.image_b64, result.ensemble_grid, alpha=req.alpha, colormap=req.colormap
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"overlay generation failed: {exc}"
        ) from exc

    return EnsembleResponse(
        model_name=backend.name,
        question=req.question,
        scales_requested=list(req.scales),
        scales_used=result.scales_used,
        scales_skipped=result.scales_skipped,
        ensemble_grid=result.ensemble_grid.astype(float).tolist(),
        per_scale=[
            PerScaleResult(scale=s, attention_grid=g.astype(float).tolist())
            for s, g in result.per_scale
        ],
        top_regions=[TopRegion(row=r, col=c, score=sc) for r, c, sc in top],
        overlay_b64=overlay_b64,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Input-sensitivity (robustness) endpoint
# ---------------------------------------------------------------------------


class SensitivityRequest(BaseModel):
    """Request body for ``POST /explain/sensitivity``.

    Carries the same explainer knobs as :class:`ExplainRequest` (so every
    method can be probed) plus the perturbation-sweep controls.
    """

    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image (PNG/JPEG).")
    model_name: str = Field(
        "mock", description="Registered backend name (see /methods)."
    )
    method: str = Field(
        "attention",
        description=f"Explanation method. Implemented: {IMPLEMENTED_METHODS}.",
    )
    question: str = Field(
        "Where is the salient region?", description="Prompt to condition the backend."
    )
    n_samples: int = Field(
        64, ge=2, le=MAX_LIME_SAMPLES, description="LIME perturbation count."
    )
    n_segments: int = Field(
        36, ge=4, le=MAX_LIME_SEGMENTS, description="LIME superpixel count."
    )
    occlusion_grid: int = Field(
        8, ge=2, le=MAX_OCCLUSION_GRID, description="GradCAM occlusion grid side."
    )
    shap_grid: int = Field(7, ge=2, le=16, description="SHAP tile grid side.")
    shap_samples: int = Field(
        32, ge=2, le=MAX_LIME_SAMPLES, description="SHAP coalitions per tile."
    )
    sigmas: list[float] = Field(
        default_factory=lambda: list(DEFAULT_SIGMAS),
        description="Gaussian noise standard deviations to sweep, each in (0, 1].",
    )
    n_trials: int = Field(
        3,
        ge=1,
        le=MAX_SENSITIVITY_TRIALS,
        description="Perturbed samples averaged per σ.",
    )
    seed: int = Field(
        0, ge=0, description="RNG seed — identical inputs give identical results."
    )
    stability_threshold: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="stability_score at/above which is_stable is true.",
    )


class PerturbationStat(BaseModel):
    """Drift at one noise level in a sensitivity sweep."""

    model_config = ConfigDict(frozen=True)
    sigma: float
    mean_drift: float
    max_drift: float


class SensitivityResponse(BaseModel):
    """Response body for ``POST /explain/sensitivity``."""

    model_config = ConfigDict(frozen=True)
    model_name: str
    method: str
    baseline_answer: str
    stability_score: float
    is_stable: bool
    worst_sigma: float
    worst_drift: float
    per_sigma: list[PerturbationStat]
    latency_ms: float


def _validate_sigmas(sigmas: list[float]) -> tuple[float, ...]:
    """Validate the noise-level sweep; raise HTTP 400 on a bad list."""
    if not sigmas:
        raise HTTPException(status_code=400, detail="sigmas must not be empty.")
    if len(sigmas) > MAX_SENSITIVITY_SIGMAS:
        raise HTTPException(
            status_code=400,
            detail=f"at most {MAX_SENSITIVITY_SIGMAS} sigma levels allowed; got {len(sigmas)}.",
        )
    for s in sigmas:
        if not (0.0 < s <= 1.0):
            raise HTTPException(
                status_code=400, detail=f"each sigma must be in (0, 1]; got {s}."
            )
    return tuple(sigmas)


@app.post("/explain/sensitivity", response_model=SensitivityResponse)
def explain_sensitivity(req: SensitivityRequest) -> SensitivityResponse:
    """Measure how much an explanation drifts under small input perturbations.

    Sweeps seeded Gaussian noise at each requested σ, re-runs the chosen
    explainer ``n_trials`` times per σ, and reports per-σ attribution drift
    plus an aggregate stability verdict. Works for every implemented method;
    see :mod:`miru.sensitivity` for the methodology.
    """
    _validate_method(req.method)
    sigmas = _validate_sigmas(req.sigmas)
    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    t0 = time.perf_counter()
    baseline_out, baseline_grid = _run_method(req.method, backend, image_array, req)

    def saliency_fn(arr: np.ndarray) -> np.ndarray:
        _, grid = _run_method(req.method, backend, arr, req)
        return grid

    result = compute_sensitivity(
        saliency_fn,
        image_array,
        baseline_grid=baseline_grid,
        baseline_answer=baseline_out.answer,
        sigmas=sigmas,
        n_trials=req.n_trials,
        seed=req.seed,
        stability_threshold=req.stability_threshold,
    )
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    return SensitivityResponse(
        model_name=backend.name,
        method=req.method,
        baseline_answer=result.baseline_answer,
        stability_score=result.stability_score,
        is_stable=result.is_stable,
        worst_sigma=result.worst_sigma,
        worst_drift=result.worst_drift,
        per_sigma=[
            PerturbationStat(
                sigma=p.sigma, mean_drift=p.mean_drift, max_drift=p.max_drift
            )
            for p in result.per_sigma
        ],
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Dataset-level saliency analytics endpoint
# ---------------------------------------------------------------------------

MAX_DATASET_BATCH = 64  # images per /analyze/batch call


class DatasetBatchItem(BaseModel):
    """One image in a dataset analytics batch."""

    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image.")
    question: str = Field("Where is the salient region?")


class SpuriousCell(BaseModel):
    model_config = ConfigDict(frozen=True)
    row: int
    col: int
    mean_saliency: float


class DatasetAnalyticsRequest(BaseModel):
    """Request body for ``POST /analyze/batch``."""

    model_config = ConfigDict(frozen=True)
    images: list[DatasetBatchItem] = Field(
        ...,
        min_length=1,
        max_length=MAX_DATASET_BATCH,
        description=f"Per-image items. 1..{MAX_DATASET_BATCH}.",
    )
    model_name: str = Field("mock", description="Backend for all images.")
    method: str = Field(
        "attention",
        description=f"Explanation method. Implemented: {IMPLEMENTED_METHODS}.",
    )
    mean_threshold: float = Field(
        0.5,
        gt=0.0,
        lt=1.0,
        description="Mean-saliency threshold for spurious-correlation detection.",
    )
    cv_threshold: float = Field(
        0.5,
        gt=0.0,
        description="CV (std/mean) upper bound for spurious-correlation detection.",
    )
    top_k: int = Field(5, ge=1, le=64)
    n_samples: int = Field(64, ge=2, le=MAX_LIME_SAMPLES)
    n_segments: int = Field(36, ge=4, le=MAX_LIME_SEGMENTS)
    occlusion_grid: int = Field(8, ge=2, le=MAX_OCCLUSION_GRID)
    shap_grid: int = Field(7, ge=2, le=16)
    shap_samples: int = Field(32, ge=2, le=MAX_LIME_SAMPLES)


class PerImageResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    index: int
    answer: str
    confidence: float
    attention_grid: list[list[float]]


class DatasetAnalyticsResponse(BaseModel):
    """Response body for ``POST /analyze/batch``."""

    model_config = ConfigDict(frozen=True)
    model_name: str
    method: str
    n_images: int
    mean_grid: list[list[float]]
    std_grid: list[list[float]]
    cv_grid: list[list[float]]
    spurious_cells: list[SpuriousCell]
    per_image: list[PerImageResult]
    latency_ms: float


@app.post("/analyze/batch", response_model=DatasetAnalyticsResponse)
def analyze_batch(req: DatasetAnalyticsRequest) -> DatasetAnalyticsResponse:
    """Run saliency extraction over a batch of images and aggregate statistics.

    Each image is explained with the chosen method.  The resulting saliency
    grids are averaged cell-wise to produce a dataset-level heatmap.
    Cells that are both high-saliency (mean ≥ ``mean_threshold``) and
    low-variance (coefficient of variation < ``cv_threshold``) are flagged
    as spurious-correlation candidates — they likely correspond to dataset
    artefacts (watermarks, fixed borders, overlays) rather than semantically
    meaningful regions.

    Spurious detection is suppressed when ``len(images) < 3`` because
    variance estimates from very small samples are not reliable.
    """
    _validate_method(req.method)
    backend = _get_backend_or_400(req.model_name)

    t0 = time.perf_counter()
    grids: list[np.ndarray] = []
    per_image: list[PerImageResult] = []

    for idx, item in enumerate(req.images):
        image_array = _decode_to_float_array(item.image_b64)
        item_req = ExplainRequest(
            image_b64=item.image_b64,
            model_name=req.model_name,
            method=req.method,
            question=item.question,
            top_k=req.top_k,
            n_samples=req.n_samples,
            n_segments=req.n_segments,
            occlusion_grid=req.occlusion_grid,
            shap_grid=req.shap_grid,
            shap_samples=req.shap_samples,
        )
        out, saliency = _run_method(req.method, backend, image_array, item_req)
        grids.append(saliency)
        per_image.append(
            PerImageResult(
                index=idx,
                answer=out.answer,
                confidence=float(out.confidence),
                attention_grid=saliency.astype(float).tolist(),
            )
        )

    analytics = analyse_dataset(
        grids,
        mean_threshold=req.mean_threshold,
        cv_threshold=req.cv_threshold,
    )
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    spurious = [
        SpuriousCell(row=r, col=c, mean_saliency=float(analytics.mean_grid[r, c]))
        for r, c in analytics.spurious_cells
    ]

    return DatasetAnalyticsResponse(
        model_name=backend.name,
        method=req.method,
        n_images=analytics.n_samples,
        mean_grid=analytics.mean_grid.astype(float).tolist(),
        std_grid=analytics.std_grid.astype(float).tolist(),
        cv_grid=analytics.cv_grid.astype(float).tolist(),
        spurious_cells=spurious,
        per_image=per_image,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Expert annotation alignment endpoint
# ---------------------------------------------------------------------------

MAX_MASK_DIM = 512  # each axis; prevents unbounded memory allocation

_ANNOTATION_ALIGNMENT_DESCRIPTION = (
    "Compare a model's saliency map against a human-supplied binary mask. "
    "Returns IoU, AUC-ROC, Spearman correlation, and a 'misaligned' flag "
    "('right answer, wrong reasoning') when the answer is correct but "
    f"IoU < {MISALIGN_THRESHOLD}."
)


class AnnotateRequest(BaseModel):
    """Request body for ``POST /annotate``."""

    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image (PNG/JPEG).")
    model_name: str = Field("mock", description="Registered backend name.")
    method: str = Field(
        "attention",
        description=f"Explanation method. Implemented: {IMPLEMENTED_METHODS}.",
    )
    question: str = Field(
        "Where is the salient region?", description="Prompt to condition the backend."
    )
    mask: list[list[float]] = Field(
        ...,
        description=(
            "Binary ground-truth mask as a 2-D list of 0/1 values. "
            f"Each axis ≤ {MAX_MASK_DIM} pixels."
        ),
    )
    answer_correct: bool = Field(
        False,
        description=(
            "Set to true when the model's answer matches your expected answer. "
            "Used to compute the 'misaligned' flag."
        ),
    )
    top_pct: float = Field(
        0.20,
        gt=0.0,
        lt=1.0,
        description="Fraction of saliency pixels used as threshold for IoU.",
    )
    top_k: int = Field(
        5, ge=1, le=64, description="Top regions returned in the explain block."
    )
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    colormap: str = Field("jet")
    n_samples: int = Field(64, ge=2, le=MAX_LIME_SAMPLES)
    n_segments: int = Field(36, ge=4, le=MAX_LIME_SEGMENTS)
    occlusion_grid: int = Field(8, ge=2, le=MAX_OCCLUSION_GRID)
    shap_grid: int = Field(7, ge=2, le=16)
    shap_samples: int = Field(32, ge=2, le=MAX_LIME_SAMPLES)


class AlignmentBlock(BaseModel):
    """Alignment scores between the saliency map and the human mask."""

    model_config = ConfigDict(frozen=True)
    iou: float
    auc_roc: float
    spearman_r: float
    top_pct: float
    misaligned: bool


class AnnotateResponse(BaseModel):
    """Response body for ``POST /annotate``."""

    model_config = ConfigDict(frozen=True)
    model_name: str
    method: str
    answer: str
    confidence: float
    overlay_b64: str
    attention_grid: list[list[float]]
    top_regions: list[TopRegion]
    alignment: AlignmentBlock
    latency_ms: float


@app.post("/annotate", response_model=AnnotateResponse)
def annotate(req: AnnotateRequest) -> AnnotateResponse:
    """Run one explanation then score it against a human-annotated mask.

    The *mask* is a 2-D list of 0/1 floats matching the annotated region.
    It does not need to match the attention grid resolution — alignment
    metrics handle the resampling internally.

    The ``misaligned`` flag in the response is set when ``answer_correct=true``
    and the spatial IoU is below the misalignment threshold
    (``MISALIGN_THRESHOLD = 0.3``), indicating "right answer, wrong reasoning".
    """
    _validate_method(req.method)
    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    mask_arr = _validate_mask(req.mask)

    t0 = time.perf_counter()
    out, saliency_grid = _run_method(req.method, backend, image_array, req)
    alignment = compare_annotation(
        saliency_grid,
        mask_arr,
        answer_correct=req.answer_correct,
        top_pct=req.top_pct,
    )
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    top = _EXTRACTOR.top_k_regions(saliency_grid, k=req.top_k)
    try:
        overlay_b64 = generate_overlay(
            req.image_b64, saliency_grid, alpha=req.alpha, colormap=req.colormap
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"overlay generation failed: {exc}"
        ) from exc

    return AnnotateResponse(
        model_name=backend.name,
        method=req.method,
        answer=out.answer,
        confidence=float(out.confidence),
        overlay_b64=overlay_b64,
        attention_grid=saliency_grid.astype(float).tolist(),
        top_regions=[TopRegion(row=r, col=c, score=s) for r, c, s in top],
        alignment=AlignmentBlock(
            iou=alignment.iou,
            auc_roc=alignment.auc_roc,
            spearman_r=alignment.spearman_r,
            top_pct=alignment.top_pct,
            misaligned=alignment.misaligned,
        ),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Cross-modal trace endpoint
# ---------------------------------------------------------------------------


class TraceRequest(BaseModel):
    """Request body for ``POST /trace``."""

    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image (PNG/JPEG).")
    model_name: str = Field("mock", description="Registered backend name.")
    question: str = Field(
        "What is the main subject of this image?",
        description="Natural-language question; tokenised on whitespace to produce word rows.",
    )


class TraceResponse(BaseModel):
    """Response body for ``POST /trace``."""

    model_config = ConfigDict(frozen=True)
    model_name: str
    question: str
    words: list[str]
    matrix: list[list[float]]
    grid_h: int
    grid_w: int
    full_attention: list[list[float]]
    latency_ms: float


_CROSS_MODAL_TRACER = CrossModalTracer()


@app.post("/trace", response_model=TraceResponse)
def trace(req: TraceRequest) -> TraceResponse:
    """Cross-modal word → image-region attribution.

    For each whitespace token in *question*, the response matrix contains
    one row of ``grid_h × grid_w`` float values in ``[0, 1]`` representing
    how much the model's spatial attention is attributed to that word.

    The attribution is computed via perturbation: removing word ``w_i`` from
    the question and measuring the positive shift in the attention map.  This
    is backend-agnostic and requires no gradients.

    ``full_attention`` is the baseline attention map for the unmodified
    question, returned so clients can render both the per-word heatmaps and
    the global heatmap from one call.
    """
    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    t0 = time.perf_counter()
    result = _CROSS_MODAL_TRACER.trace(backend, image_array, req.question)
    latency_ms = (time.perf_counter() - t0) * 1_000.0

    return TraceResponse(
        model_name=backend.name,
        question=req.question,
        words=result.words,
        matrix=result.matrix.tolist(),
        grid_h=result.grid_h,
        grid_w=result.grid_w,
        full_attention=result.full_attention.tolist(),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Batch + cache endpoints
# ---------------------------------------------------------------------------


@app.post("/explain/batch", response_model=BatchExplainResponse)
def explain_batch(req: BatchExplainRequest) -> BatchExplainResponse:
    """Run /explain over a list of images sequentially.

    Cache hits are served instantly per item; misses run end-to-end and
    populate the cache for next time. Each slot in the response is
    independent — one bad item doesn't fail the batch unless
    ``stop_on_error=true``.

    Aggregate stats roll up over the successful items only:
    ``mean_confidence`` and (when ``fidelity=true``) ``mean_fidelity``.
    """
    items: list[BatchItemResult] = []
    aborted = False
    success_confs: list[float] = []
    success_fids: list[float] = []
    cache_hits = 0
    cache_misses = 0
    total_t0 = time.perf_counter()

    for idx, item_req in enumerate(req.items):
        if aborted:
            items.append(
                BatchItemResult(
                    index=idx,
                    success=False,
                    cached=False,
                    error="skipped: prior item failed (stop_on_error=true)",
                )
            )
            continue
        try:
            if is_cache_enabled():
                resp, was_hit = _run_explain_with_cache(
                    item_req,
                    fidelity=req.fidelity,
                    synergy=req.synergy,
                    record=req.record,
                )
            else:
                resp = ExplainResponse(
                    **_run_explain_uncached(
                        item_req,
                        fidelity=req.fidelity,
                        synergy=req.synergy,
                        record=req.record,
                    )
                )
                was_hit = False
            if was_hit:
                cache_hits += 1
            else:
                cache_misses += 1
            success_confs.append(float(resp.confidence))
            if resp.fidelity is not None:
                success_fids.append(float(resp.fidelity.fidelity_score))
            items.append(
                BatchItemResult(
                    index=idx,
                    success=True,
                    cached=was_hit,
                    response=resp,
                )
            )
        except HTTPException as exc:
            items.append(
                BatchItemResult(
                    index=idx,
                    success=False,
                    cached=False,
                    error=str(exc.detail),
                )
            )
            if req.stop_on_error:
                aborted = True
        except (ValueError, RuntimeError, KeyError) as exc:
            # Boundary code: these are the realistic failure modes from
            # the image decode / backend / numpy paths. Anything else
            # is a programmer bug and should propagate.
            items.append(
                BatchItemResult(
                    index=idx,
                    success=False,
                    cached=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if req.stop_on_error:
                aborted = True

    total_latency_ms = (time.perf_counter() - total_t0) * 1_000.0
    success_count = sum(1 for it in items if it.success)
    failure_count = len(items) - success_count

    aggregate = BatchAggregate(
        total=len(items),
        success_count=success_count,
        failure_count=failure_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        mean_confidence=(sum(success_confs) / len(success_confs))
        if success_confs
        else None,
        mean_fidelity=(sum(success_fids) / len(success_fids)) if success_fids else None,
        total_latency_ms=total_latency_ms,
    )
    return BatchExplainResponse(items=items, aggregate=aggregate)


@app.get("/explain/cache_stats", response_model=CacheStatsResponse)
def explain_cache_stats() -> CacheStatsResponse:
    """Report cache hit/miss counts, entry count, and on-disk size.

    When ``MIRU_CACHE_ENABLED=0`` the response is ``{"enabled": false}``
    with every other field at its default.
    """
    cache = get_cache()
    if cache is None:
        return CacheStatsResponse(enabled=False)
    stats = cache.stats()
    return CacheStatsResponse(**stats)


@app.post("/explain/cache_clear", response_model=CacheClearResponse)
def explain_cache_clear() -> CacheClearResponse:
    """Drop every cache entry and reset the hit/miss counters.

    Returns the number of rows deleted.  A no-op (``cleared=0``) when
    the cache is disabled.
    """
    cache = get_cache()
    if cache is None:
        return CacheClearResponse(cleared=0)
    return CacheClearResponse(cleared=cache.clear())


# ---------------------------------------------------------------------------
# History + calibration + diff endpoints
# ---------------------------------------------------------------------------


@app.get("/explain/history", response_model=HistoryResponse)
def explain_history(
    limit: int = Query(
        50,
        ge=1,
        le=MAX_HISTORY_LIMIT,
        description=f"Page size, 1..{MAX_HISTORY_LIMIT}.",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Records to skip before the page starts.",
    ),
    method: str | None = Query(
        None,
        description="Exact-match explanation method filter (attention / lime / gradcam / shap).",
    ),
    model: str | None = Query(
        None,
        description="Exact-match backend name filter.",
    ),
    min_confidence: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Lower bound on trace.confidence.",
    ),
    since: str | None = Query(
        None,
        description="ISO-8601 timestamp; records strictly older are excluded.",
    ),
) -> HistoryResponse:
    """Paginated, filtered listing of past explanations, newest first.

    Reads the recorder JSONL store. The ``attention_grid`` and
    ``top_regions`` arrays are stripped from each row — fetch the full
    payload via ``/analysis/{id}/export?format=json`` when needed.
    """
    try:
        page = query_records(
            method=method,
            model=model,
            min_confidence=min_confidence,
            since=since,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return HistoryResponse(
        items=[
            HistoryItem(
                analysis_id=item.analysis_id,
                ts=item.ts,
                question=item.question,
                image_sha256=item.image_sha256,
                backend=item.backend,
                method=item.method,
                confidence=item.confidence,
                latency_ms=item.latency_ms,
                fidelity_score=item.fidelity_score,
                cache_hit=item.cache_hit,
            )
            for item in page.items
        ],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@app.get("/explain/calibration", response_model=CalibrationResponse)
def explain_calibration(
    n_bins: int = Query(
        10,
        ge=2,
        le=MAX_CALIBRATION_BINS,
        description=f"Equal-width bins on [0, 1]. 2..{MAX_CALIBRATION_BINS}.",
    ),
    method: str | None = Query(
        None,
        description="Restrict to one explanation method.",
    ),
    model: str | None = Query(
        None,
        description="Restrict to one backend.",
    ),
    limit: int = Query(
        100,
        ge=1,
        le=MAX_HISTORY_LIMIT,
        description=(
            "Maximum recent records to include in the curve. "
            "Only records carrying a fidelity score count toward this cap."
        ),
    ),
) -> CalibrationResponse:
    """Expected Calibration Error + reliability curve from recorded fidelity.

    Pulls the most recent ``limit`` records matching the filter,
    keeps only those with a fidelity score (run ``/explain?fidelity=true``
    to populate), bins by ``confidence`` into ``n_bins`` equal-width
    buckets on ``[0, 1]``, and reports ECE = Σ (n_b/N) · |conf_b − fid_b|.

    Empty population (no records with fidelity yet) returns ``ece=0.0``
    and ``n=0`` rather than 404 — clients render an "insufficient data"
    state.
    """
    # Pull a generous slice of history filtered by method/model, then
    # let compute_calibration filter to records with fidelity.
    try:
        page = query_records(
            method=method,
            model=model,
            limit=MAX_HISTORY_LIMIT,
            offset=0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Cap to ``limit`` records that actually carry a fidelity score.
    candidates = [item for item in page.items if item.fidelity_score is not None][
        :limit
    ]
    result = compute_calibration(candidates, n_bins=n_bins)

    return CalibrationResponse(
        n=result.n,
        n_bins=result.n_bins,
        ece=result.ece,
        mean_confidence=result.mean_confidence,
        mean_fidelity=result.mean_fidelity,
        bins=[
            CalibrationBinModel(
                lo=b.lo,
                hi=b.hi,
                count=b.count,
                mean_confidence=b.mean_confidence,
                mean_fidelity=b.mean_fidelity,
                gap=b.gap,
            )
            for b in result.bins
        ],
        filter_method=method,
        filter_model=model,
    )


@app.post("/explain/diff", response_model=DiffResponse)
def explain_diff(req: DiffRequest) -> DiffResponse:
    """Diff two recorded explanations by their analysis_id.

    Loads each record from the recorder store, aligns their
    attention grids (bilinearly upsampling the smaller one), computes
    cosine similarity / L2 distance / signed delta grid / top-N
    changed cells, and ships a short human-readable summary of where
    the attention shifted.

    Returns 404 when either ID is missing from the store.
    """
    if req.analysis_id_a == req.analysis_id_b:
        raise HTTPException(
            status_code=400,
            detail="analysis_id_a and analysis_id_b must differ.",
        )
    rec_a = find_record_by_id(req.analysis_id_a)
    if rec_a is None:
        raise HTTPException(
            status_code=404,
            detail=f"analysis_id '{req.analysis_id_a}' not found",
        )
    rec_b = find_record_by_id(req.analysis_id_b)
    if rec_b is None:
        raise HTTPException(
            status_code=404,
            detail=f"analysis_id '{req.analysis_id_b}' not found",
        )

    try:
        result = diff_records(rec_a, rec_b, top_n=req.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DiffResponse(
        analysis_id_a=result.analysis_id_a,
        analysis_id_b=result.analysis_id_b,
        method_a=result.method_a,
        method_b=result.method_b,
        backend_a=result.backend_a,
        backend_b=result.backend_b,
        cosine_similarity=result.cosine_similarity,
        l2_distance=result.l2_distance,
        delta_grid=result.delta_grid,
        top_changed=[
            TopChangedRegionModel(
                row=t.row,
                col=t.col,
                value_a=t.value_a,
                value_b=t.value_b,
                delta=t.delta,
            )
            for t in result.top_changed
        ],
        summary=result.summary,
    )


# ---------------------------------------------------------------------------
# Model-comparison / post-hoc consensus / search endpoints
# ---------------------------------------------------------------------------


@app.get("/explain/models/compare", response_model=ModelsCompareResponse)
def explain_models_compare(
    models: str = Query(
        ...,
        description=(
            "Comma-separated list of model names to compare "
            f"(1..{MAX_COMPARE_MODELS}). Distinct."
        ),
        examples=["mock", "mock,clip"],
    ),
    limit: int = Query(
        50,
        ge=1,
        le=MAX_HISTORY_LIMIT,
        description=f"Per-model record cap, 1..{MAX_HISTORY_LIMIT}.",
    ),
    method: str | None = Query(
        None,
        description="Optional explanation-method filter applied to every model.",
    ),
) -> ModelsCompareResponse:
    """Aggregate per-model stats over recorded history.

    Returns count / mean confidence / mean latency / mean fidelity /
    ECE / method distribution for each requested model, plus three
    winner verdicts (by confidence, fidelity, ECE).  Each metric's
    winner is ``None`` when no model has data for it.
    """
    parsed = [m.strip() for m in models.split(",") if m.strip()]
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="models must contain at least one non-empty name",
        )
    if len(parsed) > MAX_COMPARE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"too many models; max {MAX_COMPARE_MODELS}, got {len(parsed)}",
        )

    try:
        result = compare_models(parsed, limit=limit, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ModelsCompareResponse(
        models=result.models,
        stats={
            name: ModelStatsBlock(
                model=st.model,
                n_records=st.n_records,
                mean_confidence=st.mean_confidence,
                mean_latency_ms=st.mean_latency_ms,
                mean_fidelity=st.mean_fidelity,
                n_with_fidelity=st.n_with_fidelity,
                ece=st.ece,
                method_distribution=st.method_distribution,
            )
            for name, st in result.stats.items()
        },
        winner_by_confidence=result.winner_by_confidence,
        winner_by_fidelity=result.winner_by_fidelity,
        winner_by_ece=result.winner_by_ece,
        filter_method=result.filter_method,
        limit=result.limit,
    )


@app.post("/explain/consensus/by_ids", response_model=PosthocConsensusResponse)
def explain_consensus_by_ids(
    req: PosthocConsensusRequest,
) -> PosthocConsensusResponse:
    """Build a weighted-average consensus from existing analysis records.

    Loads every ID in ``analysis_ids`` from the recorder store and
    combines their attention grids via fidelity-, confidence-, or
    uniform-weighted averaging.  Each contributing record receives
    an ``agreement_score`` ∈ [-1, 1] (cosine between its grid and the
    consensus).

    Distinct from :func:`explain_consensus` (``POST /explain/consensus``)
    which takes a fresh image plus a list of methods and runs every
    method live.  This one combines analyses that have already run.

    Returns 400 on duplicate IDs / bad weighting, 404 on a missing
    record.
    """
    if len(set(req.analysis_ids)) != len(req.analysis_ids):
        raise HTTPException(
            status_code=400,
            detail="analysis_ids must be distinct",
        )

    records: list[dict] = []
    for aid in req.analysis_ids:
        rec = find_record_by_id(aid)
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail=f"analysis_id '{aid}' not found",
            )
        records.append(rec)

    try:
        result = build_posthoc_consensus(
            records,
            weighting=req.weighting,
            top_k=req.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PosthocConsensusResponse(
        consensus_grid=result.consensus_grid,
        per_record=[
            PerRecordContributionModel(
                analysis_id=p.analysis_id,
                method=p.method,
                backend=p.backend,
                weight=p.weight,
                agreement_score=p.agreement_score,
            )
            for p in result.per_record
        ],
        top_regions=[
            TopConsensusRegionModel(row=t.row, col=t.col, score=t.score)
            for t in result.top_regions
        ],
        weighting=result.weighting,
        n_records=result.n_records,
        grid_h=result.grid_h,
        grid_w=result.grid_w,
    )


@app.post("/explain/search", response_model=SearchResponse)
def explain_search(req: SearchRequest) -> SearchResponse:
    """Find recorded explanations similar to the query attribution grid.

    Two query modes:

    - Supply ``query_grid`` directly — a 2-D float saliency map.
    - Supply ``query_analysis_id`` — its stored grid becomes the query
      and that same record is excluded from the result set.

    Filters: ``method`` and ``model``. Optional ``min_similarity``
    cutoff. Bilinearly aligns candidate grids to the query's shape so
    differently-resolved methods can be compared.

    Returns 400 on bad arguments; 404 on missing query_analysis_id.
    """
    try:
        result = search_by_pattern(
            query_grid=req.query_grid,
            query_analysis_id=req.query_analysis_id,
            method=req.method,
            model=req.model,
            top_k=req.top_k,
            min_similarity=req.min_similarity,
            max_scan=req.max_scan,
        )
    except ValueError as exc:
        # Disambiguate "not found" from other validation errors so
        # clients can return 404 to their own users.
        message = str(exc)
        if "not found in store" in message:
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc

    return SearchResponse(
        matches=[
            SearchMatchModel(
                analysis_id=m.analysis_id,
                ts=m.ts,
                method=m.method,
                backend=m.backend,
                question=m.question,
                similarity=m.similarity,
            )
            for m in result.matches
        ],
        n_candidates=result.n_candidates,
        n_scanned=result.n_scanned,
        top_k=result.top_k,
        query_analysis_id=result.query_analysis_id,
    )


# ---------------------------------------------------------------------------
# Alert rules + history endpoints
# ---------------------------------------------------------------------------

MAX_ALERT_HISTORY_LIMIT = 200


class CreateRuleRequest(BaseModel):
    """Body for ``POST /explain/alerts/rules``."""

    model_config = ConfigDict(frozen=True)
    name: str = Field(
        ..., min_length=1, max_length=200, description="Human-readable rule name."
    )
    field: str = Field(
        ...,
        description=f"Condition field. One of: {SUPPORTED_FIELDS}.",
    )
    op: str = Field(
        ...,
        description=f"Comparison operator. One of: {SUPPORTED_OPS}.",
    )
    threshold: float = Field(..., description="Numeric threshold for the condition.")
    webhook_url: str = Field(
        ...,
        description="HTTP/HTTPS URL that receives a POST when the rule fires.",
    )


class RuleResponse(BaseModel):
    """One rule row serialised for the API."""

    model_config = ConfigDict(frozen=True)
    rule_id: str
    name: str
    field: str
    op: str
    threshold: float
    webhook_url: str
    enabled: bool
    created_at: str


class RulesListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    rules: list[RuleResponse]
    total: int


class DeleteRuleResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    deleted: bool
    rule_id: str


class FiredAlertResponse(BaseModel):
    """One fired-alert history row."""

    model_config = ConfigDict(frozen=True)
    alert_id: str
    rule_id: str
    rule_name: str
    analysis_id: str
    field: str
    fired_value: float
    threshold: float
    op: str
    webhook_url: str
    ts: str
    delivered: bool


class AlertHistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    alerts: list[FiredAlertResponse]
    total: int
    limit: int


@app.post("/explain/alerts/rules", response_model=RuleResponse, status_code=201)
def create_alert_rule(req: CreateRuleRequest) -> RuleResponse:
    """Create a new alert rule.

    The rule fires whenever ``POST /explain`` produces an output where
    ``field op threshold`` is true (e.g. ``confidence < 0.4``).
    When fired, a POST is delivered to ``webhook_url`` and the event is
    recorded in alert history regardless of delivery status.

    Returns 400 on validation errors (unknown field/op, bad URL).
    Returns 503 when the alert subsystem is disabled
    (``MIRU_ALERTS_ENABLED=0``).
    """
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="alert subsystem is disabled (MIRU_ALERTS_ENABLED=0)",
        )
    if req.field not in SUPPORTED_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"field must be one of {list(SUPPORTED_FIELDS)}, got {req.field!r}",
        )
    if req.op not in SUPPORTED_OPS:
        raise HTTPException(
            status_code=400,
            detail=f"op must be one of {list(SUPPORTED_OPS)}, got {req.op!r}",
        )
    try:
        validate_webhook_url(req.webhook_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        rule = store.create_rule(
            name=req.name,
            field=req.field,
            op=req.op,
            threshold=req.threshold,
            webhook_url=req.webhook_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RuleResponse(**rule.to_dict())


@app.get("/explain/alerts/rules", response_model=RulesListResponse)
def list_alert_rules(
    enabled_only: bool = Query(
        False, description="When true, return only enabled rules."
    ),
) -> RulesListResponse:
    """List all alert rules."""
    store = get_store()
    if store is None:
        return RulesListResponse(rules=[], total=0)
    rules = store.list_rules(enabled_only=enabled_only)
    return RulesListResponse(
        rules=[RuleResponse(**r.to_dict()) for r in rules],
        total=len(rules),
    )


@app.delete("/explain/alerts/rules/{rule_id}", response_model=DeleteRuleResponse)
def delete_alert_rule(
    rule_id: str = FastApiPath(
        ..., min_length=8, description="rule_id returned by POST /explain/alerts/rules"
    ),
) -> DeleteRuleResponse:
    """Delete one alert rule by ID. Returns 404 when the ID is not found."""
    store = get_store()
    if store is None:
        raise HTTPException(status_code=503, detail="alert subsystem is disabled")
    deleted = store.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"rule_id '{rule_id}' not found")
    return DeleteRuleResponse(deleted=True, rule_id=rule_id)


@app.get("/explain/alerts/history", response_model=AlertHistoryResponse)
def alert_history(
    limit: int = Query(
        50,
        ge=1,
        le=MAX_ALERT_HISTORY_LIMIT,
        description=f"Maximum rows to return (1..{MAX_ALERT_HISTORY_LIMIT}).",
    ),
) -> AlertHistoryResponse:
    """Return recent fired alerts, newest first."""
    store = get_store()
    if store is None:
        return AlertHistoryResponse(alerts=[], total=0, limit=limit)
    alerts = store.list_alerts(limit=limit)
    return AlertHistoryResponse(
        alerts=[FiredAlertResponse(**a.to_dict()) for a in alerts],
        total=len(alerts),
        limit=limit,
    )


__all__ = ["app"]
