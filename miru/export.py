"""Export a recorded analysis to PNG / JSON / PDF.

The recorder writes privacy-stripped JSONL records keyed by
``analysis_id``.  This module turns one record into a portable artefact
suitable for archival or audit handover.

Output formats
--------------

- **JSON**: the full recorded record (raw bytes returned via the API,
  with content-type ``application/json``).
- **PNG**:  the saliency map colorised through the jet palette and
  upsampled 2× via nearest-neighbour to a 32×32 (or whatever the
  source resolution × 2 produces) display image.  No source image is
  composited — the recorder never persists pixels, and the export
  honours that boundary.
- **PDF**:  a single-page document containing the PNG above with a
  metadata header strip (timestamp, question, model, fidelity score).
  Falls back to PNG-bytes when Pillow is unavailable.

All three paths go through ``export_record(record, format)`` which
returns ``(bytes, content_type, suggested_filename)``.
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np

from miru.visualization.overlay import attention_to_heatmap, encode_png_b64

DEFAULT_FORMAT = "json"
SUPPORTED_FORMATS: tuple[str, ...] = ("png", "json", "pdf")
PNG_SCALE = 2  # display upsample factor


def export_record(
    record: dict[str, Any], fmt: str = DEFAULT_FORMAT
) -> tuple[bytes, str, str]:
    """Render a recorded analysis to the requested format.

    Args:
        record: A JSONL record from :func:`miru.recorder.build_record`.
        fmt: One of ``"png"``, ``"json"``, ``"pdf"``.

    Returns:
        ``(payload_bytes, content_type, suggested_filename)``.

    Raises:
        ValueError: When ``fmt`` is not in :data:`SUPPORTED_FORMATS`.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"format must be one of {SUPPORTED_FORMATS}, got {fmt!r}"
        )
    analysis_id = record.get("analysis_id", "analysis")
    if fmt == "json":
        payload = json.dumps(record, indent=2, sort_keys=False).encode("utf-8")
        return payload, "application/json", f"{analysis_id}.json"
    if fmt == "png":
        png_bytes = _heatmap_png_bytes(record)
        return png_bytes, "image/png", f"{analysis_id}.png"
    # fmt == "pdf"
    pdf_bytes, content_type, filename = _record_pdf_bytes(record, analysis_id)
    return pdf_bytes, content_type, filename


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _attention_grid(record: dict[str, Any]) -> np.ndarray:
    """Pull the attention/saliency grid out of the recorded trace.

    Looks under both ``trace.attention_map.data`` (the legacy
    ``ReasoningTrace`` shape) and ``trace.attention_grid`` (the api/main
    ExplainResponse shape).
    """
    trace = record.get("trace") or {}
    am = trace.get("attention_map")
    if isinstance(am, dict) and am.get("data") is not None:
        return np.asarray(am["data"], dtype=np.float32)
    grid = trace.get("attention_grid")
    if grid is not None:
        return np.asarray(grid, dtype=np.float32)
    # Last-ditch fallback: a 1×1 zero grid so the export path never
    # crashes on a malformed record.
    return np.zeros((1, 1), dtype=np.float32)


def _heatmap_png_bytes(record: dict[str, Any]) -> bytes:
    """Colorise the saliency grid and emit PNG bytes at 2× scale."""
    grid = _attention_grid(record)
    rgba = attention_to_heatmap(grid, colormap="jet")  # (H, W, 4) uint8
    upsampled = _nearest_upsample(rgba, PNG_SCALE)
    b64 = encode_png_b64(upsampled)
    # encode_png_b64 returns base64; decode to raw bytes for the wire.
    import base64

    return base64.b64decode(b64)


def _nearest_upsample(rgba: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upsample an (H, W, C) uint8 array by integer scale."""
    if scale <= 1:
        return rgba
    return np.repeat(np.repeat(rgba, scale, axis=0), scale, axis=1)


def _record_pdf_bytes(
    record: dict[str, Any], analysis_id: str
) -> tuple[bytes, str, str]:
    """Render a single-page PDF: header strip + heatmap.

    Uses Pillow's ``Image.save(format="PDF")`` so we don't pull in a
    new dependency.  If Pillow is unavailable, fall back to handing the
    PNG bytes back with a PDF filename (clients that asked for PDF
    still get a viewable artefact)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        png = _heatmap_png_bytes(record)
        return png, "image/png", f"{analysis_id}.pdf.png"

    grid = _attention_grid(record)
    rgba = attention_to_heatmap(grid, colormap="jet")
    upsampled = _nearest_upsample(rgba, max(PNG_SCALE * 8, PNG_SCALE))

    heat_img = Image.fromarray(upsampled, mode="RGBA").convert("RGB")
    HW = heat_img.width
    header_h = max(80, HW // 5)
    page_w = HW + 80
    page_h = header_h + HW + 80

    page = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
    # Header strip
    draw = ImageDraw.Draw(page)
    draw.rectangle([0, 0, page_w, header_h], fill=(20, 22, 32))
    draw.rectangle([0, header_h, page_w, header_h + 2], fill=(176, 140, 255))

    trace = record.get("trace") or {}
    fidelity = trace.get("fidelity") or {}
    fid_score = fidelity.get("fidelity_score") if isinstance(fidelity, dict) else fidelity

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None  # Pillow always has the default font; guard anyway.

    lines = [
        f"Miru analysis · {analysis_id}",
        f"timestamp · {record.get('ts', 'unknown')}",
        f"question  · {record.get('question', '')[:80]}",
        f"answer    · {str(trace.get('answer', ''))[:80]}",
        f"backend   · {trace.get('backend', 'unknown')}   "
        f"method · {trace.get('method', trace.get('explanation_method', 'unknown'))}   "
        f"confidence · {trace.get('confidence', 'n/a')}   "
        f"fidelity · {fid_score if fid_score is not None else 'n/a'}",
    ]
    y = 12
    for line in lines:
        draw.text((20, y), line, fill=(238, 240, 247), font=font)
        y += 14

    # Heatmap image
    page.paste(heat_img, (40, header_h + 40))

    buf = io.BytesIO()
    page.save(buf, format="PDF")
    return buf.getvalue(), "application/pdf", f"{analysis_id}.pdf"


__all__ = [
    "DEFAULT_FORMAT",
    "SUPPORTED_FORMATS",
    "PNG_SCALE",
    "export_record",
]
