"""Tests for miru.annotation and the POST /annotate endpoint."""
from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_b64(width: int = 16, height: int = 16) -> str:
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFF_FFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode()


def _grid(size: int = 16, val: float = 0.5) -> np.ndarray:
    return np.full((size, size), val, dtype=np.float32)


def _hot_grid(size: int = 16) -> np.ndarray:
    """Attention concentrated in top-left quadrant."""
    g = np.zeros((size, size), dtype=np.float32)
    half = size // 2
    g[:half, :half] = 1.0
    return g


def _mask_top_left(size: int = 16) -> np.ndarray:
    """Ground-truth mask covering top-left quadrant."""
    m = np.zeros((size, size), dtype=np.float32)
    half = size // 2
    m[:half, :half] = 1.0
    return m


def _mask_bottom_right(size: int = 16) -> np.ndarray:
    """Ground-truth mask covering bottom-right quadrant — opposite of hot_grid."""
    m = np.zeros((size, size), dtype=np.float32)
    half = size // 2
    m[half:, half:] = 1.0
    return m


# ---------------------------------------------------------------------------
# Unit — compare_annotation
# ---------------------------------------------------------------------------


def test_perfect_alignment_iou_is_one() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_top_left()
    result = compare_annotation(sal, mask)
    assert result.iou == pytest.approx(1.0, abs=0.05)


def test_perfect_alignment_auc_near_one() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_top_left()
    result = compare_annotation(sal, mask)
    assert result.auc_roc > 0.9


def test_inverted_alignment_iou_is_zero() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_bottom_right()
    result = compare_annotation(sal, mask)
    assert result.iou == pytest.approx(0.0, abs=0.05)


def test_spearman_perfect_positive() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_top_left()
    result = compare_annotation(sal, mask)
    assert result.spearman_r > 0.5


def test_spearman_negative_when_inverted() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_bottom_right()
    result = compare_annotation(sal, mask)
    assert result.spearman_r < 0.0


def test_spearman_near_zero_for_uniform_saliency() -> None:
    from miru.annotation import compare_annotation

    sal = _grid(val=0.5)
    mask = _mask_top_left()
    result = compare_annotation(sal, mask)
    assert abs(result.spearman_r) < 0.1


def test_top_pct_preserved_in_result() -> None:
    from miru.annotation import compare_annotation

    result = compare_annotation(_hot_grid(), _mask_top_left(), top_pct=0.10)
    assert result.top_pct == pytest.approx(0.10)


def test_misaligned_false_when_answer_wrong_and_low_iou() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_bottom_right()
    result = compare_annotation(sal, mask, answer_correct=False)
    assert result.misaligned is False


def test_misaligned_true_when_correct_answer_low_iou() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_bottom_right()  # IoU ~ 0 — well below threshold
    result = compare_annotation(sal, mask, answer_correct=True)
    assert result.misaligned is True


def test_misaligned_false_when_correct_answer_high_iou() -> None:
    from miru.annotation import compare_annotation

    sal = _hot_grid()
    mask = _mask_top_left()  # perfect overlap
    result = compare_annotation(sal, mask, answer_correct=True)
    assert result.misaligned is False


def test_different_resolution_mask_does_not_raise() -> None:
    from miru.annotation import compare_annotation

    sal = np.random.default_rng(0).random((8, 8)).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    result = compare_annotation(sal, mask)
    assert 0.0 <= result.iou <= 1.0
    assert 0.0 <= result.auc_roc <= 1.0


def test_non_2d_saliency_raises() -> None:
    from miru.annotation import compare_annotation

    with pytest.raises(ValueError, match="saliency must be 2-D"):
        compare_annotation(np.zeros((4, 4, 2)), np.zeros((4, 4)))


def test_non_2d_mask_raises() -> None:
    from miru.annotation import compare_annotation

    with pytest.raises(ValueError, match="mask must be 2-D"):
        compare_annotation(np.zeros((4, 4)), np.zeros((4,)))


def test_empty_mask_raises() -> None:
    from miru.annotation import compare_annotation

    with pytest.raises(ValueError, match="empty"):
        compare_annotation(np.zeros((4, 4)), np.zeros((0, 0)))


def test_invalid_top_pct_raises() -> None:
    from miru.annotation import compare_annotation

    with pytest.raises(ValueError, match="top_pct"):
        compare_annotation(_hot_grid(), _mask_top_left(), top_pct=1.5)


def test_return_values_in_valid_ranges() -> None:
    from miru.annotation import compare_annotation

    rng = np.random.default_rng(7)
    sal = rng.random((16, 16)).astype(np.float32)
    mask = (rng.random((16, 16)) > 0.7).astype(np.float32)
    result = compare_annotation(sal, mask)
    assert 0.0 <= result.iou <= 1.0
    assert 0.0 <= result.auc_roc <= 1.0
    assert -1.0 <= result.spearman_r <= 1.0


# ---------------------------------------------------------------------------
# Unit — _spearman helper
# ---------------------------------------------------------------------------


def test_spearman_helper_perfect_agreement() -> None:
    from miru.annotation import _spearman

    sal = np.array([[0.0, 0.5], [0.5, 1.0]])
    mask = np.array([[0.0, 0.0], [0.0, 1.0]])
    rho = _spearman(sal, mask)
    assert rho > 0.5


def test_spearman_helper_constant_returns_zero() -> None:
    from miru.annotation import _spearman

    sal = np.ones((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)
    assert _spearman(sal, mask) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# API integration — POST /annotate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    from miru.models import registry

    registry.register_defaults()
    return TestClient(app)


def _make_mask(size: int = 16, region: str = "top_left") -> list[list[float]]:
    m = [[0.0] * size for _ in range(size)]
    half = size // 2
    if region == "top_left":
        for r in range(half):
            for c in range(half):
                m[r][c] = 1.0
    elif region == "all":
        for r in range(size):
            for c in range(size):
                m[r][c] = 1.0
    return m


def test_annotate_happy_path_returns_200(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "Where is the subject?",
        "mask": _make_mask(),
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 200, resp.text


def test_annotate_response_has_alignment_block(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "test",
        "mask": _make_mask(),
    }
    body = api_client.post("/annotate", json=payload).json()
    al = body["alignment"]
    assert "iou" in al and "auc_roc" in al and "spearman_r" in al
    assert "top_pct" in al and "misaligned" in al


def test_annotate_alignment_values_in_range(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "Where?",
        "mask": _make_mask(),
    }
    al = api_client.post("/annotate", json=payload).json()["alignment"]
    assert 0.0 <= al["iou"] <= 1.0
    assert 0.0 <= al["auc_roc"] <= 1.0
    assert -1.0 <= al["spearman_r"] <= 1.0


def test_annotate_misaligned_false_when_answer_wrong(api_client) -> None:
    # misaligned is only set when answer_correct=True AND IoU is low.
    # answer_correct defaults to False so misaligned must be False regardless.
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "anything",
        "mask": _make_mask(),
        "answer_correct": False,
    }
    al = api_client.post("/annotate", json=payload).json()["alignment"]
    assert al["misaligned"] is False


def test_annotate_response_includes_explain_fields(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "test",
        "mask": _make_mask(),
    }
    body = api_client.post("/annotate", json=payload).json()
    assert "answer" in body
    assert "confidence" in body
    assert "overlay_b64" in body
    assert "attention_grid" in body
    assert "top_regions" in body
    assert body["latency_ms"] >= 0.0


def test_annotate_unknown_model_returns_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "nope_xyz",
        "method": "attention",
        "question": "test",
        "mask": _make_mask(),
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 400
    assert "nope_xyz" in resp.json()["detail"]


def test_annotate_bad_image_returns_400(api_client) -> None:
    payload = {
        "image_b64": "!!!not_base64",
        "model_name": "mock",
        "method": "attention",
        "question": "test",
        "mask": _make_mask(),
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 400


def test_annotate_empty_mask_returns_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "test",
        "mask": [],
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 400


def test_annotate_jagged_mask_returns_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "test",
        "mask": [[0.0, 1.0], [0.0]],
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 400


def test_annotate_oversized_mask_returns_400(api_client) -> None:
    big = [[0.0] * 513 for _ in range(513)]
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "test",
        "mask": big,
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 400


def test_annotate_unknown_method_returns_400(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "magic",
        "question": "test",
        "mask": _make_mask(),
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 400


def test_annotate_lime_method_works(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "lime",
        "question": "test lime",
        "mask": _make_mask(),
        "n_samples": 4,
        "n_segments": 4,
    }
    resp = api_client.post("/annotate", json=payload)
    assert resp.status_code == 200, resp.text


def test_annotate_top_pct_respected(api_client) -> None:
    payload = {
        "image_b64": _png_b64(),
        "model_name": "mock",
        "method": "attention",
        "question": "pct",
        "mask": _make_mask(),
        "top_pct": 0.10,
    }
    body = api_client.post("/annotate", json=payload).json()
    assert body["alignment"]["top_pct"] == pytest.approx(0.10, abs=1e-6)


def test_annotate_health_not_regressed(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
