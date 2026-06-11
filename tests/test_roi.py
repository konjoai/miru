"""Unit tests for BoundingBox validation and ROI grid-embedding math."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.main import BoundingBox


# ---------------------------------------------------------------------------
# BoundingBox model validation
# ---------------------------------------------------------------------------


def test_bounding_box_valid() -> None:
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    assert bbox.x1 == pytest.approx(0.1)
    assert bbox.y2 == pytest.approx(0.9)


def test_bounding_box_rejects_x1_equal_x2() -> None:
    with pytest.raises(ValidationError, match="x2"):
        BoundingBox(x1=0.5, y1=0.0, x2=0.5, y2=1.0)


def test_bounding_box_rejects_x2_less_than_x1() -> None:
    with pytest.raises(ValidationError, match="x2"):
        BoundingBox(x1=0.7, y1=0.0, x2=0.3, y2=1.0)


def test_bounding_box_rejects_y1_equal_y2() -> None:
    with pytest.raises(ValidationError, match="y2"):
        BoundingBox(x1=0.0, y1=0.5, x2=1.0, y2=0.5)


def test_bounding_box_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x1=-0.1, y1=0.0, x2=0.5, y2=1.0)
    with pytest.raises(ValidationError):
        BoundingBox(x1=0.0, y1=0.0, x2=1.1, y2=1.0)


def test_bounding_box_full_image() -> None:
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    assert bbox.x2 == pytest.approx(1.0)
