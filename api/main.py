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
from pydantic import BaseModel, ConfigDict, Field

from miru import gradcam_explainer, lime_explainer
from miru.shap_explainer import SHAPConfig, SHAPExplainer
from miru.attention.extractor import AttentionExtractor
from miru.bench.comparison import compare_backends
from miru.bench.runner import run_benchmark
from miru.config import settings
from miru.consensus import compute_consensus
from miru.eu_ai_act import generate_report as generate_eu_ai_act_report
from miru.export import SUPPORTED_FORMATS, export_record
from miru.fidelity import LOW_FIDELITY_THRESHOLD, deletion_test
from miru.models import registry
from miru.recorder import find_record_by_id, maybe_record
from miru.visualization.overlay import decode_image_b64, generate_overlay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits — bounded to keep a public deployment honest and predictable.
# ---------------------------------------------------------------------------

MAX_BENCH_N = 100        # cap synth-bench sample count per request
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


class ExplainRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    image_b64: str = Field(..., description="Base64-encoded source image (PNG/JPEG).")
    model_name: str = Field("mock", description="Registered backend name (see /methods).")
    method: str = Field("attention", description=f"Explanation method. Implemented: {IMPLEMENTED_METHODS}.")
    question: str = Field("Where is the salient region?", description="Prompt to condition the backend.")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Heatmap opacity for the overlay.")
    colormap: str = Field("jet", description="One of jet | hot | viridis.")
    top_k: int = Field(5, ge=1, le=64, description="Number of top attention regions to return.")
    n_samples: int = Field(64, ge=2, le=MAX_LIME_SAMPLES, description="LIME perturbation count.")
    n_segments: int = Field(36, ge=4, le=MAX_LIME_SEGMENTS, description="LIME superpixel count.")
    occlusion_grid: int = Field(8, ge=2, le=MAX_OCCLUSION_GRID, description="GradCAM occlusion grid side.")
    shap_grid: int = Field(7, ge=2, le=16, description="SHAP tile grid side (shap_grid × shap_grid tiles).")
    shap_samples: int = Field(32, ge=2, le=MAX_LIME_SAMPLES, description="SHAP coalitions sampled per tile.")


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
    model_name: str = Field("mock", description="Backend name (shared by both methods).")
    method_a: str = Field("attention", description="First explanation method.")
    method_b: str = Field("gradcam", description="Second explanation method.")
    question: str = Field("Where is the salient region?")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    colormap: str = Field("jet")
    top_k: int = Field(5, ge=1, le=64)
    n_samples: int = Field(48, ge=2, le=MAX_LIME_SAMPLES)
    n_segments: int = Field(36, ge=4, le=MAX_LIME_SEGMENTS)
    shap_grid: int = Field(5, ge=2, le=16, description="SHAP tile grid side.")
    shap_samples: int = Field(16, ge=2, le=MAX_LIME_SAMPLES, description="SHAP coalitions per tile.")
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


@app.post("/explain", response_model=ExplainResponse)
def explain(
    req: ExplainRequest,
    fidelity: bool = Query(
        default=False,
        description=(
            "Run the deletion test on the saliency map and include a "
            "fidelity block in the response. Doubles the backend call "
            "count, so off by default."
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
) -> ExplainResponse:
    _validate_method(req.method)
    backend = _get_backend_or_400(req.model_name)
    image_array = _decode_to_float_array(req.image_b64)

    t0 = time.perf_counter()
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

    fidelity_block: FidelityBlock | None = None
    if fidelity:
        result = deletion_test(
            backend, image_array, req.question, saliency_grid,
            baseline_confidence=float(out.confidence),
        )
        fidelity_block = FidelityBlock(
            fidelity_score=result.fidelity_score,
            baseline_confidence=result.baseline_confidence,
            masked_confidence=result.masked_confidence,
            k_pct=result.k_pct,
            low_fidelity=result.low_fidelity,
        )

    # Build a stable trace dict for the recorder. Stripped of overlay
    # by build_record; we drop the attention_grid here too — it's
    # bulky and the export path re-derives it from the original record
    # via the attention_grid field (kept) so the user can still PNG/PDF.
    trace_dict = {
        "answer": out.answer,
        "confidence": float(out.confidence),
        "backend": backend.name,
        "method": req.method,
        "explanation_method": req.method,
        "latency_ms": latency_ms,
        "attention_grid": saliency_grid.astype(float).tolist(),
        "top_regions": [{"row": r, "col": c, "score": s} for r, c, s in top],
        "fidelity": fidelity_block.model_dump() if fidelity_block else None,
    }
    analysis_id = (
        maybe_record(trace_dict, image_b64=req.image_b64, question=req.question)
        if record else None
    ) or ""

    return ExplainResponse(
        model_name=backend.name,
        method=req.method,
        answer=out.answer,
        confidence=float(out.confidence),
        overlay_b64=overlay_b64,
        attention_grid=saliency_grid.astype(float).tolist(),
        top_regions=[TopRegion(row=r, col=c, score=s) for r, c, s in top],
        latency_ms=latency_ms,
        analysis_id=analysis_id,
        fidelity=fidelity_block,
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
        overlay_a = generate_overlay(req.image_b64, sal_a, alpha=req.alpha, colormap=req.colormap)
        overlay_b = generate_overlay(req.image_b64, sal_b, alpha=req.alpha, colormap=req.colormap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"overlay generation failed: {exc}") from exc

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
        per_method.append((
            method,
            sal,
            MethodResult(
                method=method,
                overlay_b64=overlay,
                attention_grid=sal.astype(float).tolist(),
                top_regions=[TopRegion(row=r, col=c, score=s) for r, c, s in top],
            ),
        ))

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
    Article 13 (transparency), and Article 15 (accuracy & robustness).

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
            detail=(
                f"format must be one of {SUPPORTED_FORMATS}, got {format!r}"
            ),
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
                f"unknown model_name='{model_name}'. "
                f"Available: {registry.available()}."
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


def _float_array_to_pil(image_array: np.ndarray) -> "Image.Image":
    """Convert a float32 (H, W, 3) ∈ [0, 1] array to a PIL RGB Image."""
    from PIL import Image

    uint8 = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


__all__ = ["app"]
