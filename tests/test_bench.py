"""Tests for the saliency benchmark harness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from miru.bench.metrics import auc_roc, bilinear_upsample, hit_at_k, iou_at_topk_pct
from miru.bench.runner import (
    SCHEMA_VERSION,
    compare_results,
    load_result,
    run_benchmark,
    save_result,
)
from miru.bench.synth import (
    SynthSample,
    generate_dataset,
    generate_sample,
)
from miru.cli import build_parser, main
from miru.cli.bench import run_compare, run_run, run_show


# ===========================================================================
# Synth — determinism & shape contracts
# ===========================================================================


def test_synth_sample_shape_and_dtype() -> None:
    s = generate_sample(seed=1, index=0, size=32)
    assert s.image.shape == (32, 32, 3)
    assert s.image.dtype == np.float32
    assert s.image.min() >= 0.0 and s.image.max() <= 1.0
    assert s.mask.shape == (32, 32)
    assert s.mask.dtype == np.bool_
    assert s.mask.any(), "ground-truth mask must mark at least one pixel"


def test_synth_is_deterministic() -> None:
    a = generate_sample(seed=7, index=4)
    b = generate_sample(seed=7, index=4)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.mask, b.mask)
    assert a.question == b.question
    assert a.meta == b.meta


def test_synth_different_index_different_output() -> None:
    a = generate_sample(seed=7, index=0)
    b = generate_sample(seed=7, index=1)
    assert not np.array_equal(a.image, b.image)


def test_synth_variant_cycle() -> None:
    variants = [generate_sample(seed=42, index=i).meta["variant"] for i in range(6)]
    # 3 variants → 6 samples cover 2 full cycles in fixed order.
    assert variants == ["single", "two", "low_snr", "single", "two", "low_snr"]


def test_synth_two_variant_has_two_centroids() -> None:
    s = generate_sample(seed=42, index=1)  # index 1 → "two"
    assert s.meta["variant"] == "two"
    assert len(s.meta["centroids"]) == 2


def test_synth_mask_centred_on_blob() -> None:
    """The mask centroid (computed from mask pixels) should match the recorded centroid."""
    s = generate_sample(seed=3, index=0, size=64)  # single-blob variant
    ys, xs = np.where(s.mask)
    obs_cy = float(ys.mean())
    obs_cx = float(xs.mean())
    rec_cy, rec_cx = s.meta["centroids"][0]
    assert abs(obs_cy - rec_cy) < 1.5
    assert abs(obs_cx - rec_cx) < 1.5


def test_generate_dataset_size() -> None:
    ds = generate_dataset(seed=0, n=10, size=32)
    assert len(ds) == 10
    assert all(isinstance(s, SynthSample) for s in ds)


# ===========================================================================
# Metrics — edge cases that pin the math
# ===========================================================================


def test_bilinear_upsample_identity() -> None:
    g = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = bilinear_upsample(g, 2, 2)
    assert np.array_equal(out, g)


def test_bilinear_upsample_size_match() -> None:
    g = np.random.default_rng(0).random((4, 4)).astype(np.float32)
    out = bilinear_upsample(g, 16, 16)
    assert out.shape == (16, 16)
    # Corner values must be preserved.
    assert abs(float(out[0, 0]) - float(g[0, 0])) < 1e-6
    assert abs(float(out[-1, -1]) - float(g[-1, -1])) < 1e-6


def test_iou_perfect() -> None:
    """Attention identical to mask → IoU == 1.0 at any positive top_pct."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 2:7] = True
    attn = mask.astype(np.float32)
    iou = iou_at_topk_pct(attn, mask, top_pct=mask.mean())
    assert iou == pytest.approx(1.0, abs=1e-6)


def test_iou_zero_when_disjoint() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:3, 0:3] = True
    attn = np.zeros((10, 10), dtype=np.float32)
    attn[7:10, 7:10] = 1.0
    iou = iou_at_topk_pct(attn, mask, top_pct=0.09)
    assert iou == 0.0


def test_iou_validates_top_pct() -> None:
    mask = np.ones((4, 4), dtype=bool)
    attn = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        iou_at_topk_pct(attn, mask, top_pct=0.0)
    with pytest.raises(ValueError):
        iou_at_topk_pct(attn, mask, top_pct=1.0)


def test_auc_perfect_inverted_random() -> None:
    """AUC: perfect = 1.0, perfectly inverted = 0.0, random ≈ 0.5."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    perfect = mask.astype(np.float32)
    inverted = 1.0 - perfect

    assert auc_roc(perfect, mask) == pytest.approx(1.0, abs=1e-6)
    assert auc_roc(inverted, mask) == pytest.approx(0.0, abs=1e-6)

    rng = np.random.default_rng(0)
    random_attn = rng.random((20, 20)).astype(np.float32)
    auc = auc_roc(random_attn, mask)
    assert 0.35 < auc < 0.65, f"random AUC drifted: {auc}"


def test_auc_degenerate_mask_returns_chance() -> None:
    mask_all = np.ones((4, 4), dtype=bool)
    mask_none = np.zeros((4, 4), dtype=bool)
    attn = np.random.default_rng(0).random((4, 4)).astype(np.float32)
    assert auc_roc(attn, mask_all) == 0.5
    assert auc_roc(attn, mask_none) == 0.5


def test_hit_at_k_argmax_inside() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2, 3] = True
    attn = np.zeros((8, 8), dtype=np.float32)
    attn[2, 3] = 1.0
    assert hit_at_k(attn, mask, k=1) == 1.0


def test_hit_at_k_argmax_outside() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 0] = True
    attn = np.zeros((8, 8), dtype=np.float32)
    attn[7, 7] = 1.0
    assert hit_at_k(attn, mask, k=1) == 0.0


def test_hit_at_k_resamples_mask_to_attn_shape() -> None:
    """Mask at higher resolution still scores correctly at attn resolution."""
    attn = np.zeros((4, 4), dtype=np.float32)
    attn[1, 2] = 1.0  # argmax cell in attn
    mask = np.zeros((16, 16), dtype=bool)
    # The (1, 2) cell of a 4x4 attn corresponds to row centres ~6, col centres ~10.
    mask[6:8, 9:12] = True
    assert hit_at_k(attn, mask, k=1) == 1.0


def test_hit_at_k_validates_k() -> None:
    with pytest.raises(ValueError):
        hit_at_k(np.zeros((4, 4)), np.zeros((4, 4), dtype=bool), k=0)


# ===========================================================================
# Runner — schema & behaviour
# ===========================================================================


def test_run_benchmark_smoke() -> None:
    result = run_benchmark("mock", n=4, seed=11, size=32)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["backend"] == "mock"
    assert result["n"] == 4
    assert len(result["samples"]) == 4
    for key in ("iou", "auc", "hit1", "latency_ms"):
        assert key in result["aggregate"]
        for stat in ("mean", "std", "p50", "p95", "n"):
            assert stat in result["aggregate"][key]


def test_run_benchmark_unknown_backend_falls_back() -> None:
    result = run_benchmark("does_not_exist", n=2, seed=1, size=32)
    assert result["backend"] == "mock"


def test_run_benchmark_metrics_in_range() -> None:
    result = run_benchmark("mock", n=6, seed=3, size=32)
    for s in result["samples"]:
        assert 0.0 <= s["iou"] <= 1.0
        assert 0.0 <= s["auc"] <= 1.0
        assert 0.0 <= s["hit1"] <= 1.0
        assert s["latency_ms"] > 0.0


def test_run_benchmark_hardware_metadata_present() -> None:
    result = run_benchmark("mock", n=2, seed=1, size=32)
    hw = result["hardware"]
    for key in ("platform", "python", "machine", "numpy"):
        assert key in hw and hw[key]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    result = run_benchmark("mock", n=3, seed=5, size=32)
    path = save_result(result, tmp_path / "run.json")
    loaded = load_result(path)
    assert loaded["aggregate"]["iou"]["mean"] == result["aggregate"]["iou"]["mean"]
    assert len(loaded["samples"]) == len(result["samples"])


def test_compare_results_paired_delta(tmp_path: Path) -> None:
    a = run_benchmark("mock", n=5, seed=9, size=32)
    b = run_benchmark("mock", n=5, seed=9, size=32)
    cmp = compare_results(a, b, metric="iou")
    # Same backend + same seed → identical samples → zero delta.
    assert cmp["mean_delta"] == pytest.approx(0.0, abs=1e-9)
    assert cmp["n"] == 5
    assert cmp["a_mean"] == cmp["b_mean"]


def test_compare_results_rejects_unpaired() -> None:
    a = run_benchmark("mock", n=4, seed=1, size=32)
    b = run_benchmark("mock", n=4, seed=2, size=32)  # different seed
    with pytest.raises(ValueError):
        compare_results(a, b, metric="iou")


# ===========================================================================
# CLI
# ===========================================================================


def test_cli_parser_accepts_bench_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["bench", "run", "--backend", "mock", "--n", "5"])
    assert (args.cmd, args.bench_cmd, args.backend, args.n) == ("bench", "run", "mock", 5)
    args = parser.parse_args(["bench", "show", "/tmp/x.json"])
    assert (args.cmd, args.bench_cmd, args.path) == ("bench", "show", "/tmp/x.json")
    args = parser.parse_args(["bench", "compare", "/tmp/a.json", "/tmp/b.json", "--metric", "auc"])
    assert (args.cmd, args.bench_cmd, args.metric) == ("bench", "compare", "auc")


def test_cli_run_writes_result_file(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    code = main([
        "bench", "run", "--backend", "mock",
        "--n", "3", "--seed", "1", "--out", str(out),
    ])
    assert code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["n"] == 3
    assert data["backend"] == "mock"


def test_cli_show_loads_saved_run(tmp_path: Path, capsys) -> None:
    out = tmp_path / "run.json"
    main(["bench", "run", "--backend", "mock", "--n", "2", "--seed", "1", "--out", str(out)])
    capsys.readouterr()  # discard run output

    code = main(["bench", "show", str(out)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Miru bench" in captured
    assert "backend=mock" in captured


def test_cli_compare_two_runs(tmp_path: Path, capsys) -> None:
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    main(["bench", "run", "--backend", "mock", "--n", "4", "--seed", "7", "--out", str(a_path)])
    main(["bench", "run", "--backend", "mock", "--n", "4", "--seed", "7", "--out", str(b_path)])
    capsys.readouterr()

    code = main(["bench", "compare", str(a_path), str(b_path), "--metric", "iou"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "compare iou" in captured
    assert "tie" in captured  # identical seeds → no delta
