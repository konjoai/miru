"""``miru export …`` subcommand — generate HTML report + PNG tiles from a bench result."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def run_export_report(
    in_path: str,
    out_dir: str,
    *,
    alpha: float = 0.50,
    colormap: str = "jet",
    no_mask_border: bool = False,
    no_png_tiles: bool = False,
    stream=None,
) -> int:
    """Generate an HTML report + optional PNG tiles from a saved benchmark result.

    Args:
        in_path:        Path to a saved benchmark JSON (from ``miru bench run``).
        out_dir:        Output directory; created if absent.
        alpha:          Heatmap opacity (0 = invisible, 1 = opaque).
        colormap:       Heatmap colormap — "jet", "hot", or "viridis".
        no_mask_border: When True, suppress the GT mask boundary overlay.
        no_png_tiles:   When True, skip writing individual PNG tile files.
        stream:         Output stream (defaults to sys.stdout).

    Returns:
        Exit code — 0 on success, 1 on error.
    """
    out = stream or sys.stdout
    from miru.bench.export import generate_report
    from miru.bench.runner import load_result

    out.write(f"loading {in_path}\n")
    try:
        result = load_result(in_path)
    except (OSError, ValueError) as exc:
        out.write(f"error: cannot load result — {exc}\n")
        return 1

    out.write(
        f"exporting: backend={result['backend']} n={result['n']} seed={result['seed']}"
        f" → {out_dir}\n"
    )
    try:
        report_path = generate_report(
            result,
            out_dir,
            alpha=alpha,
            colormap=colormap,
            show_mask_border=not no_mask_border,
            write_png_tiles=not no_png_tiles,
        )
    except Exception as exc:  # noqa: BLE001
        out.write(f"error: export failed — {exc}\n")
        return 1

    n_tiles = result["n"] * 2 if not no_png_tiles else 0  # raw + overlay per sample
    out.write(f"wrote {report_path}\n")
    if not no_png_tiles:
        out.write(f"wrote {n_tiles} PNG tiles to {out_dir}/\n")
    return 0


__all__ = ["run_export_report"]
