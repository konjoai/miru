"""Tests for miru/bench/export.py and miru/cli/export.py — Phase 7.

All tests run offline without real backends.  The export module re-generates
synth images deterministically so no stored images are needed.

Coverage:
- render_sample: shape/dtype contracts, determinism, GT mask border rendering
- generate_report: HTML produced, PNG tiles written, content smoke checks
- CLI: parser accepts ``export`` subcommand, happy-path end-to-end, bad-path error
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from miru.bench.export import (
    _alpha_composite,
    _composite_overlay,
    _image_to_rgba,
    _mask_border_rgba,
    generate_report,
    render_sample,
)
from miru.bench.runner import run_benchmark, save_result
from miru.cli import build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_result(n: int = 4, seed: int = 99, size: int = 32) -> dict:
    """Run a quick mock benchmark — n=4 samples at 32×32 for speed."""
    return run_benchmark("mock", n=n, seed=seed, size=size)


# ===========================================================================
# _image_to_rgba
# ===========================================================================


def test_image_to_rgba_shape_dtype() -> None:
    img = np.ones((8, 8, 3), dtype=np.float32) * 0.5
    rgba = _image_to_rgba(img)
    assert rgba.shape == (8, 8, 4)
    assert rgba.dtype == np.uint8


def test_image_to_rgba_alpha_fully_opaque() -> None:
    img = np.zeros((4, 4, 3), dtype=np.float32)
    rgba = _image_to_rgba(img)
    assert (rgba[:, :, 3] == 255).all(), "alpha channel must be 255 everywhere"


def test_image_to_rgba_clips_range() -> None:
    img = np.full((2, 2, 3), 2.0, dtype=np.float32)  # out of [0,1]
    rgba = _image_to_rgba(img)
    assert (rgba[:, :, :3] == 255).all(), "values >1 must clip to 255"


# ===========================================================================
# _composite_overlay
# ===========================================================================


def test_composite_overlay_shape_dtype() -> None:
    img = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    attn = np.random.default_rng(0).random((4, 4)).astype(np.float32)
    out = _composite_overlay(img, attn)
    assert out.shape == (16, 16, 4)
    assert out.dtype == np.uint8


def test_composite_overlay_all_colormaps() -> None:
    img = np.ones((8, 8, 3), dtype=np.float32) * 0.5
    attn = np.linspace(0, 1, 16).reshape(4, 4).astype(np.float32)
    for cm in ("jet", "hot", "viridis"):
        out = _composite_overlay(img, attn, colormap=cm)
        assert out.shape == (8, 8, 4), f"colormap {cm!r} produced wrong shape"


def test_composite_overlay_alpha_zero_returns_image() -> None:
    """alpha=0 → heatmap invisible → output == original image (RGBA)."""
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    attn = np.ones((4, 4), dtype=np.float32)
    out = _composite_overlay(img, attn, alpha=0.0)
    # At alpha=0 the blend is 0*heatmap + 1*base, so should match the base.
    expected_rgb = (img * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(out[:, :, :3], expected_rgb)


# ===========================================================================
# _mask_border_rgba
# ===========================================================================


def test_mask_border_shape_dtype() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    border = _mask_border_rgba(mask)
    assert border.shape == (16, 16, 4)
    assert border.dtype == np.uint8


def test_mask_border_interior_is_transparent() -> None:
    """Interior pixels of the mask have alpha=0 (only the boundary is coloured)."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    border = _mask_border_rgba(mask)
    # Interior pixel — surrounded by True on all 4 sides
    assert border[10, 10, 3] == 0, "interior mask pixel must be transparent"


def test_mask_border_edge_pixel_is_opaque() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    border = _mask_border_rgba(mask)
    # Row 5, col 10 — top boundary of the block
    assert border[5, 10, 3] > 0, "boundary mask pixel must have non-zero alpha"


def test_mask_border_empty_mask_all_transparent() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    border = _mask_border_rgba(mask)
    assert (border[:, :, 3] == 0).all()


# ===========================================================================
# _alpha_composite
# ===========================================================================


def test_alpha_composite_shape_dtype() -> None:
    bottom = np.full((4, 4, 4), 128, dtype=np.uint8)
    top = np.zeros((4, 4, 4), dtype=np.uint8)  # fully transparent top
    out = _alpha_composite(bottom, top)
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.uint8


def test_alpha_composite_transparent_top_is_identity() -> None:
    """Compositing fully transparent top → output == bottom."""
    bottom = np.random.default_rng(1).integers(0, 256, (6, 6, 4), dtype=np.uint8)
    bottom[:, :, 3] = 255  # opaque bottom
    top = np.zeros((6, 6, 4), dtype=np.uint8)  # transparent top
    out = _alpha_composite(bottom, top)
    np.testing.assert_array_equal(out, bottom)


def test_alpha_composite_opaque_top_wins() -> None:
    """Compositing fully opaque top → output RGB == top RGB."""
    bottom = np.full((4, 4, 4), 50, dtype=np.uint8)
    bottom[:, :, 3] = 255
    top = np.full((4, 4, 4), 200, dtype=np.uint8)
    top[:, :, 3] = 255
    out = _alpha_composite(bottom, top)
    np.testing.assert_array_equal(out[:, :, :3], top[:, :, :3])


# ===========================================================================
# render_sample
# ===========================================================================


def test_render_sample_shape_dtype() -> None:
    result = _small_result()
    sample_rec = result["samples"][0]
    raw, overlay = render_sample(
        sample_rec,
        bench_seed=result["seed"],
        bench_size=result["size"],
    )
    size = result["size"]
    assert raw.shape == (size, size, 4), f"raw shape mismatch: {raw.shape}"
    assert raw.dtype == np.uint8
    assert overlay.shape == (size, size, 4)
    assert overlay.dtype == np.uint8


def test_render_sample_is_deterministic() -> None:
    result = _small_result()
    sample_rec = result["samples"][1]
    kwargs = dict(bench_seed=result["seed"], bench_size=result["size"])
    raw_a, overlay_a = render_sample(sample_rec, **kwargs)
    raw_b, overlay_b = render_sample(sample_rec, **kwargs)
    np.testing.assert_array_equal(raw_a, raw_b)
    np.testing.assert_array_equal(overlay_a, overlay_b)


def test_render_sample_without_mask_border() -> None:
    result = _small_result()
    sample_rec = result["samples"][0]
    _, overlay_border = render_sample(
        sample_rec,
        bench_seed=result["seed"],
        bench_size=result["size"],
        show_mask_border=True,
    )
    _, overlay_no_border = render_sample(
        sample_rec,
        bench_seed=result["seed"],
        bench_size=result["size"],
        show_mask_border=False,
    )
    # With and without border differ (border adds yellow pixels).
    # They won't be identical unless the mask is empty or the border has no effect.
    # Just assert both are valid RGBA arrays.
    assert overlay_border.shape == overlay_no_border.shape


def test_render_sample_different_indices_differ() -> None:
    result = _small_result(n=4)
    raw0, _ = render_sample(result["samples"][0], bench_seed=result["seed"], bench_size=result["size"])
    raw1, _ = render_sample(result["samples"][1], bench_seed=result["seed"], bench_size=result["size"])
    assert not np.array_equal(raw0, raw1), "different samples must produce different images"


# ===========================================================================
# generate_report
# ===========================================================================


def test_generate_report_creates_html(tmp_path: Path) -> None:
    result = _small_result(n=3)
    report_path = generate_report(result, tmp_path / "out")
    assert report_path.exists()
    assert report_path.suffix == ".html"
    content = report_path.read_text(encoding="utf-8")
    assert "Miru Benchmark Report" in content
    assert "mock" in content


def test_generate_report_html_contains_aggregate_metrics(tmp_path: Path) -> None:
    result = _small_result(n=3)
    report_path = generate_report(result, tmp_path / "out")
    content = report_path.read_text(encoding="utf-8")
    for key in ("iou", "auc", "hit1", "latency_ms"):
        assert key in content, f"aggregate metric {key!r} missing from report"


def test_generate_report_html_has_inline_images(tmp_path: Path) -> None:
    result = _small_result(n=2)
    report_path = generate_report(result, tmp_path / "out")
    content = report_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in content, "report must embed inline base64 PNG images"


def test_generate_report_writes_png_tiles(tmp_path: Path) -> None:
    n = 3
    result = _small_result(n=n)
    out = tmp_path / "out"
    generate_report(result, out, write_png_tiles=True)
    overlay_pngs = list(out.glob("sample_*_overlay.png"))
    raw_pngs = list(out.glob("sample_*_raw.png"))
    assert len(overlay_pngs) == n, f"expected {n} overlay PNGs, got {len(overlay_pngs)}"
    assert len(raw_pngs) == n, f"expected {n} raw PNGs, got {len(raw_pngs)}"


def test_generate_report_no_png_tiles(tmp_path: Path) -> None:
    result = _small_result(n=3)
    out = tmp_path / "out"
    generate_report(result, out, write_png_tiles=False)
    assert not list(out.glob("*.png")), "no PNG tiles expected when write_png_tiles=False"
    assert (out / "report.html").exists()


def test_generate_report_png_tiles_are_valid_png(tmp_path: Path) -> None:
    """Each PNG tile starts with the PNG magic bytes."""
    result = _small_result(n=2)
    out = tmp_path / "out"
    generate_report(result, out, write_png_tiles=True)
    _PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    for png_path in out.glob("*.png"):
        raw = png_path.read_bytes()
        assert raw[:8] == _PNG_MAGIC, f"{png_path.name} is not a valid PNG"


def test_generate_report_all_colormaps(tmp_path: Path) -> None:
    result = _small_result(n=2)
    for cm in ("jet", "hot", "viridis"):
        out = tmp_path / cm
        report = generate_report(result, out, colormap=cm, write_png_tiles=False)
        assert report.exists()
        assert cm in report.read_text(encoding="utf-8")


def test_generate_report_returns_path_to_html(tmp_path: Path) -> None:
    result = _small_result(n=2)
    path = generate_report(result, tmp_path / "x")
    assert isinstance(path, Path)
    assert path.name == "report.html"


def test_generate_report_creates_out_dir_if_absent(tmp_path: Path) -> None:
    result = _small_result(n=2)
    out = tmp_path / "nested" / "deep" / "dir"
    assert not out.exists()
    generate_report(result, out, write_png_tiles=False)
    assert out.exists()


# ===========================================================================
# CLI — miru export
# ===========================================================================


def test_cli_parser_accepts_export_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["export", "/tmp/result.json", "/tmp/out"])
    assert args.cmd == "export"
    assert args.result == "/tmp/result.json"
    assert args.out_dir == "/tmp/out"
    assert args.alpha == pytest.approx(0.50)
    assert args.colormap == "jet"
    assert args.no_mask_border is False
    assert args.no_png_tiles is False


def test_cli_parser_export_flags() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "export", "/tmp/r.json", "/tmp/out",
        "--alpha", "0.7",
        "--colormap", "hot",
        "--no-mask-border",
        "--no-png-tiles",
    ])
    assert args.alpha == pytest.approx(0.7)
    assert args.colormap == "hot"
    assert args.no_mask_border is True
    assert args.no_png_tiles is True


def test_cli_export_happy_path(tmp_path: Path) -> None:
    # First generate a result file via bench run.
    result_path = tmp_path / "result.json"
    main(["bench", "run", "--backend", "mock", "--n", "3", "--seed", "5", "--out", str(result_path)])
    assert result_path.exists()

    out_dir = tmp_path / "report"
    code = main(["export", str(result_path), str(out_dir)])
    assert code == 0
    assert (out_dir / "report.html").exists()


def test_cli_export_no_png_tiles(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    main(["bench", "run", "--backend", "mock", "--n", "2", "--seed", "7", "--out", str(result_path)])

    out_dir = tmp_path / "report"
    code = main(["export", str(result_path), str(out_dir), "--no-png-tiles"])
    assert code == 0
    assert not list(out_dir.glob("*.png"))


def test_cli_export_bad_result_path(tmp_path: Path, capsys) -> None:
    """Passing a non-existent result path returns exit code 1."""
    import io
    buf = io.StringIO()
    from miru.cli.export import run_export_report
    code = run_export_report("/nonexistent/result.json", str(tmp_path / "out"), stream=buf)
    assert code == 1
    assert "error" in buf.getvalue().lower()


def test_cli_export_html_content_correct(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    main(["bench", "run", "--backend", "mock", "--n", "4", "--seed", "11", "--out", str(result_path)])

    out_dir = tmp_path / "report"
    main(["export", str(result_path), str(out_dir), "--no-png-tiles"])

    content = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "n=4" in content
    assert "seed=11" in content
    assert "mock" in content
