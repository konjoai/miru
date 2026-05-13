"""Tests for miru/export.py (analysis exporter) + recorder analysis_id lookup.

Distinct from tests/test_export.py which covers miru/bench/export.py (the
benchmark report renderer).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miru import recorder as rec_mod
from miru.export import SUPPORTED_FORMATS, export_record
from miru.recorder import build_record, find_record_by_id, reset_recorder


def _sample_record(analysis_id: str = "abc12345-pkg-test-uuid") -> dict:
    return {
        "analysis_id": analysis_id,
        "ts": "2026-05-12T10:00:00+00:00",
        "question": "What is in the image?",
        "image_sha256": "0" * 64,
        "trace": {
            "answer": "A circular bright object.",
            "confidence": 0.83,
            "backend": "mock",
            "method": "attention",
            "latency_ms": 0.5,
            "attention_grid": [
                [0.0, 0.2, 0.4, 0.2],
                [0.2, 0.6, 0.8, 0.6],
                [0.4, 0.8, 1.0, 0.8],
                [0.2, 0.6, 0.8, 0.6],
            ],
        },
    }


# ---------------------------------------------------------------------------
# build_record + analysis_id
# ---------------------------------------------------------------------------


def test_build_record_generates_analysis_id_when_absent() -> None:
    rec = build_record({"answer": "ok"}, image_b64="abc", question="q")
    assert "analysis_id" in rec
    assert isinstance(rec["analysis_id"], str)
    assert len(rec["analysis_id"]) >= 30  # UUID v4 is 36 chars


def test_build_record_respects_supplied_analysis_id() -> None:
    rec = build_record(
        {"answer": "ok"}, image_b64="abc", question="q",
        analysis_id="custom-id-123",
    )
    assert rec["analysis_id"] == "custom-id-123"


def test_two_records_have_distinct_auto_ids() -> None:
    a = build_record({"answer": "ok"}, image_b64="a", question="q")
    b = build_record({"answer": "ok"}, image_b64="a", question="q")
    assert a["analysis_id"] != b["analysis_id"]


# ---------------------------------------------------------------------------
# find_record_by_id
# ---------------------------------------------------------------------------


def test_find_record_by_id_returns_none_for_missing(tmp_path: Path) -> None:
    assert find_record_by_id("does-not-exist", str(tmp_path)) is None


def test_find_record_by_id_scans_jsonl(tmp_path: Path) -> None:
    """Write a couple of records to disk and look one up by id."""
    a = _sample_record("first-id-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    b = _sample_record("second-id-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    f = tmp_path / "traces-20260512T100000-000000.jsonl"
    f.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    found = find_record_by_id(b["analysis_id"], str(tmp_path))
    assert found is not None
    assert found["analysis_id"] == b["analysis_id"]
    # Missing id still returns None even when the dir has matches.
    assert find_record_by_id("not-there", str(tmp_path)) is None


def test_find_record_by_id_skips_corrupt_lines(tmp_path: Path) -> None:
    a = _sample_record("survivor-id-cccc-cccc-cccc-cccccccccccc")
    f = tmp_path / "traces-20260512T100100-000000.jsonl"
    f.write_text("not-json\n" + json.dumps(a) + "\n")
    found = find_record_by_id(a["analysis_id"], str(tmp_path))
    assert found is not None


def test_find_record_by_id_empty_id_returns_none() -> None:
    assert find_record_by_id("") is None
    reset_recorder()


# ---------------------------------------------------------------------------
# export_record
# ---------------------------------------------------------------------------


def test_export_record_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        export_record(_sample_record(), fmt="xml")


def test_export_record_json_round_trips() -> None:
    payload, content_type, name = export_record(_sample_record(), "json")
    assert content_type == "application/json"
    assert name.endswith(".json")
    reconstructed = json.loads(payload)
    assert reconstructed["analysis_id"] == _sample_record()["analysis_id"]


def test_export_record_png_has_valid_signature() -> None:
    """PNG bytes begin with the 8-byte PNG file signature."""
    payload, content_type, name = export_record(_sample_record(), "png")
    assert content_type == "image/png"
    assert name.endswith(".png")
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_export_record_pdf_is_pdf_or_png_fallback() -> None:
    """PDF exporter falls back to PNG if Pillow isn't available — accept either."""
    payload, content_type, _ = export_record(_sample_record(), "pdf")
    if content_type == "application/pdf":
        assert payload.startswith(b"%PDF-")
    else:
        assert content_type == "image/png"
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_supported_formats_constant_matches_implementation() -> None:
    """SUPPORTED_FORMATS is the documented set; export_record agrees."""
    assert set(SUPPORTED_FORMATS) == {"png", "json", "pdf"}
    for fmt in SUPPORTED_FORMATS:
        payload, _, _ = export_record(_sample_record(), fmt)
        assert isinstance(payload, bytes) and len(payload) > 0


def test_build_record_output_exports_cleanly() -> None:
    """An end-to-end happy path: build → export round-trip without errors."""
    trace = {
        "answer": "x",
        "confidence": 0.6,
        "backend": "mock",
        "method": "attention",
        "attention_grid": [[0.1, 0.9], [0.2, 0.4]],
    }
    record = build_record(trace, image_b64="abc", question="q?")
    payload, _, _ = export_record(record, "json")
    data = json.loads(payload)
    assert data["trace"]["attention_grid"] == trace["attention_grid"]
    rec_mod.reset_recorder()
