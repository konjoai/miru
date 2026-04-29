"""Attention overlay visualization utilities for Miru.

Provides heatmap generation, alpha-blending, and PNG/base64 encode/decode
without requiring matplotlib.  Pillow is used when available; a minimal
pure-zlib PNG encoder is provided as fallback so the module works in
constrained CI environments.
"""
from __future__ import annotations

import base64
import io
import struct
import zlib

import numpy as np

# ---------------------------------------------------------------------------
# Optional Pillow import — used for resize and encode/decode where available.
# ---------------------------------------------------------------------------
try:
    from PIL import Image as _PILImage

    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    _PILImage = None  # type: ignore[assignment]
    _HAVE_PIL = False


# ---------------------------------------------------------------------------
# Jet colormap (piecewise linear, no matplotlib dependency)
# ---------------------------------------------------------------------------

def _jet_single(t: float) -> tuple[float, float, float]:
    """Map scalar t ∈ [0, 1] to (R, G, B) ∈ [0, 1] using the jet colormap.

    Breakpoints (inclusive):
      [0.000, 0.125]  R=0,         G=0,         B=0.5→1.0
      [0.125, 0.375]  R=0,         G=0→1,       B=1.0
      [0.375, 0.625]  R=0→1,       G=1,         B=1→0
      [0.625, 0.875]  R=1,         G=1→0,       B=0
      [0.875, 1.000]  R=1→0.5,     G=0,         B=0
    """
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.125:
        s = t / 0.125
        return 0.0, 0.0, 0.5 + 0.5 * s
    elif t <= 0.375:
        s = (t - 0.125) / 0.25
        return 0.0, s, 1.0
    elif t <= 0.625:
        s = (t - 0.375) / 0.25
        return s, 1.0, 1.0 - s
    elif t <= 0.875:
        s = (t - 0.625) / 0.25
        return 1.0, 1.0 - s, 0.0
    else:
        s = (t - 0.875) / 0.125
        return 1.0 - 0.5 * s, 0.0, 0.0


def _hot_single(t: float) -> tuple[float, float, float]:
    """Map scalar t ∈ [0, 1] to (R, G, B) using the 'hot' colormap.

    black → red → orange → yellow → white
      [0.000, 0.333]  R=0→1,   G=0,     B=0
      [0.333, 0.667]  R=1,     G=0→1,   B=0
      [0.667, 1.000]  R=1,     G=1,     B=0→1
    """
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 1 / 3:
        return t * 3.0, 0.0, 0.0
    elif t <= 2 / 3:
        return 1.0, (t - 1 / 3) * 3.0, 0.0
    else:
        return 1.0, 1.0, (t - 2 / 3) * 3.0


def _viridis_single(t: float) -> tuple[float, float, float]:
    """Map scalar t ∈ [0, 1] to (R, G, B) using a viridis approximation.

    Viridis keypoints (sampled from the canonical LUT):
      0.0 → (0.267, 0.005, 0.329)
      0.25 → (0.229, 0.322, 0.545)
      0.5  → (0.128, 0.566, 0.551)
      0.75 → (0.370, 0.788, 0.384)
      1.0  → (0.993, 0.906, 0.144)
    """
    _VIRIDIS_KEYS: list[tuple[float, float, float, float]] = [
        (0.00, 0.267004, 0.004874, 0.329415),
        (0.25, 0.229739, 0.322361, 0.545706),
        (0.50, 0.127568, 0.566949, 0.550556),
        (0.75, 0.369214, 0.788888, 0.382914),
        (1.00, 0.993248, 0.906157, 0.143936),
    ]
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(_VIRIDIS_KEYS) - 1):
        t0, r0, g0, b0 = _VIRIDIS_KEYS[i]
        t1, r1, g1, b1 = _VIRIDIS_KEYS[i + 1]
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return r0 + s * (r1 - r0), g0 + s * (g1 - g0), b0 + s * (b1 - b0)
    t0, r0, g0, b0 = _VIRIDIS_KEYS[-1]
    return r0, g0, b0


_COLORMAP_FN = {
    "jet": _jet_single,
    "hot": _hot_single,
    "viridis": _viridis_single,
}


def attention_to_heatmap(
    attention: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """Convert a normalized 2-D attention map to an RGBA uint8 array (H×W×4).

    Args:
        attention: 2-D float array with values in [0, 1].  Shape (H, W).
        colormap:  One of ``"jet"``, ``"hot"``, or ``"viridis"``.

    Returns:
        RGBA uint8 array of shape (H, W, 4).  Alpha channel is fully opaque
        (255) for every pixel.

    Raises:
        ValueError: If *attention* is not 2-D or *colormap* is unsupported.
    """
    if attention.ndim != 2:
        raise ValueError(f"attention must be 2-D, got shape {attention.shape}")
    if colormap not in _COLORMAP_FN:
        raise ValueError(f"unsupported colormap '{colormap}'; choose from {list(_COLORMAP_FN)}")

    fn = _COLORMAP_FN[colormap]
    h, w = attention.shape
    flat = attention.ravel()
    rgba = np.empty((len(flat), 4), dtype=np.float64)
    for idx, val in enumerate(flat):
        r, g, b = fn(float(val))
        rgba[idx, 0] = r
        rgba[idx, 1] = g
        rgba[idx, 2] = b
        rgba[idx, 3] = 1.0

    result = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result.reshape(h, w, 4)


def _bilinear_resize_attention(attention: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinearly upsample/downsample a 2-D float array to (target_h, target_w).

    Uses Pillow when available, otherwise falls back to nearest-neighbour via
    NumPy index mapping (adequate for attention maps at typical resolutions).
    """
    src_h, src_w = attention.shape
    if src_h == target_h and src_w == target_w:
        return attention.copy()

    if _HAVE_PIL:
        # Pass the 2-D (H×W) uint8 array directly; Pillow infers mode="L".
        grey_arr = (np.clip(attention, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = _PILImage.fromarray(grey_arr)
        resized = pil_img.resize((target_w, target_h), _PILImage.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    # Pure-NumPy nearest-neighbour fallback.
    row_idx = np.floor(np.linspace(0, src_h - 1, target_h)).astype(np.int32)
    col_idx = np.floor(np.linspace(0, src_w - 1, target_w)).astype(np.int32)
    return attention[np.ix_(row_idx, col_idx)].astype(np.float32)


def overlay_attention_on_image(
    image_rgba: np.ndarray,
    attention: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend an attention heatmap onto an RGBA image.

    The attention map is bilinearly upsampled to the image spatial dimensions
    before blending.  The jet colormap is used for the heatmap.

    Args:
        image_rgba: Original image as (H, W, 4) uint8 RGBA array.
        attention:  2-D float array with values in [0, 1], any resolution.
        alpha:      Heatmap opacity in [0, 1].  0 = invisible, 1 = opaque.

    Returns:
        (H, W, 4) uint8 RGBA array with the heatmap blended over the image.
    """
    if image_rgba.ndim != 3 or image_rgba.shape[2] != 4:
        raise ValueError(f"image_rgba must be (H, W, 4), got {image_rgba.shape}")
    if attention.ndim != 2:
        raise ValueError(f"attention must be 2-D, got shape {attention.shape}")

    h, w = image_rgba.shape[:2]
    attn_up = _bilinear_resize_attention(attention, h, w)
    heatmap = attention_to_heatmap(attn_up, colormap="jet").astype(np.float32)
    base = image_rgba.astype(np.float32)

    blended = np.clip(heatmap * alpha + base * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# PNG encode/decode
# ---------------------------------------------------------------------------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk: length + type + data + CRC."""
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def _encode_png_pure(image_rgba: np.ndarray) -> bytes:
    """Minimal pure-Python PNG encoder for RGBA uint8 arrays.

    Supports only 8-bit RGBA (colour type 6).  No ancillary chunks beyond
    IHDR/IDAT/IEND.  Compliant with PNG specification section 11.
    """
    h, w = image_rgba.shape[:2]
    png_sig = b"\x89PNG\r\n\x1a\n"

    # IHDR: width, height, bit depth, colour type (6=RGBA), compression, filter, interlace
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    ihdr = _png_chunk(b"IHDR", ihdr_data)

    # IDAT: filter-byte (0=None) prepended to each row, then zlib-compress.
    raw_rows = bytearray()
    for row in image_rgba:
        raw_rows.append(0)  # filter type None
        raw_rows.extend(row.astype(np.uint8).tobytes())

    idat = _png_chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=6))
    iend = _png_chunk(b"IEND", b"")
    return png_sig + ihdr + idat + iend


def encode_png_b64(image_rgba: np.ndarray) -> str:
    """Encode an RGBA uint8 numpy array to a base64-encoded PNG string.

    Uses Pillow when available for full PNG compliance; falls back to a
    minimal pure-Python PNG encoder otherwise.

    Args:
        image_rgba: (H, W, 4) uint8 RGBA array.

    Returns:
        Base64-encoded PNG as a plain ASCII string (no ``data:`` prefix).
    """
    if image_rgba.ndim == 2:
        # Promote greyscale to RGBA.
        grey = image_rgba[:, :, np.newaxis]
        image_rgba = np.concatenate([grey, grey, grey, np.full_like(grey, 255)], axis=2)
    if image_rgba.shape[2] == 3:
        alpha = np.full((*image_rgba.shape[:2], 1), 255, dtype=np.uint8)
        image_rgba = np.concatenate([image_rgba, alpha], axis=2)

    arr = image_rgba.astype(np.uint8)

    if _HAVE_PIL:
        # Pass the (H×W×4) uint8 array directly; Pillow infers mode="RGBA".
        pil_img = _PILImage.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # Pure fallback.
    png_bytes = _encode_png_pure(arr)
    return base64.b64encode(png_bytes).decode("ascii")


def decode_image_b64(b64_str: str) -> np.ndarray:
    """Decode a base64 image string (PNG/JPEG/etc.) to an RGBA uint8 array.

    Requires Pillow.  Returns an (H, W, 4) uint8 RGBA array.

    Args:
        b64_str: Base64-encoded image bytes (no ``data:`` prefix needed).

    Returns:
        (H, W, 4) uint8 RGBA numpy array.

    Raises:
        ImportError: If Pillow is not installed.
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    if not _HAVE_PIL:
        raise ImportError("Pillow is required for decode_image_b64; install with: pip install Pillow")

    raw = base64.b64decode(b64_str)
    try:
        pil_img = _PILImage.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Cannot decode image from base64 payload: {exc}") from exc

    return np.asarray(pil_img, dtype=np.uint8)


def generate_overlay(
    image_b64: str,
    attention: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> str:
    """Full pipeline: decode image → heatmap → overlay → encode PNG base64.

    Args:
        image_b64:  Base64-encoded source image (any format Pillow supports).
        attention:  2-D float array with values in [0, 1].
        alpha:      Heatmap opacity (0 = invisible, 1 = opaque).
        colormap:   Colormap name; passed to :func:`attention_to_heatmap`.

    Returns:
        Base64-encoded PNG string of the overlay image.
    """
    image_rgba = decode_image_b64(image_b64)

    # Resize attention to image dimensions then build the colorised heatmap.
    h, w = image_rgba.shape[:2]
    attn_up = _bilinear_resize_attention(attention, h, w)
    heatmap = attention_to_heatmap(attn_up, colormap=colormap).astype(np.float32)
    base = image_rgba.astype(np.float32)

    blended = np.clip(heatmap * alpha + base * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)
    return encode_png_b64(blended)
