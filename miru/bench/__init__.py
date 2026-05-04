"""Saliency benchmark harness — synth dataset, metrics, runner."""
from miru.bench.metrics import auc_roc, bilinear_upsample, hit_at_k, iou_at_topk_pct
from miru.bench.runner import (
    SCHEMA_VERSION,
    compare_results,
    load_result,
    run_benchmark,
    save_result,
)
from miru.bench.synth import (
    DEFAULT_SIZE,
    SynthSample,
    Variant,
    generate_dataset,
    generate_sample,
)

__all__ = [
    "SCHEMA_VERSION",
    "SynthSample",
    "Variant",
    "DEFAULT_SIZE",
    "generate_sample",
    "generate_dataset",
    "iou_at_topk_pct",
    "auc_roc",
    "hit_at_k",
    "bilinear_upsample",
    "run_benchmark",
    "save_result",
    "load_result",
    "compare_results",
]
