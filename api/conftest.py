"""Shared pytest fixtures for the deployable API test-suite."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest


@pytest.fixture
def png_b64() -> str:
    """32×32 RGB PNG with a bright 8×8 patch in the top-left.

    Pillow is used when available; otherwise the pure-zlib encoder keeps
    the fixture working in mock-only environments.
    """
    h = w = 32
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    img[4:12, 4:12] = (240, 200, 100)
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        from miru.visualization.overlay import encode_png_b64

        rgba = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        return encode_png_b64(rgba)
