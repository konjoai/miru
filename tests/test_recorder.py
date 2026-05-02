"""Tests for miru/recorder.py — privacy, threading, fsspec backend."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miru import recorder as rec_mod
from miru.recorder import (
    TraceRecorder,
    build_record,
    hash_image,
    is_recording_enabled,
    maybe_record,
    reset_recorder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_dir(tmp_path, monkeypatch):
    """Enable recording into a fresh tmp directory; reset singleton afterwards."""
    target = tmp_path / "miru_traces"
    monkeypatch.setenv("MIRU_RECORD", "1")
    monkeypatch.setenv("MIRU_RECORD_PATH", str(target))
    reset_recorder()
    yield target
    reset_recorder()


# ---------------------------------------------------------------------------
# Privacy & record shape
# ---------------------------------------------------------------------------


def test_hash_image_is_sha256_hex() -> None:
    h = hash_image("hello")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    assert h == hash_image("hello")  # deterministic


def test_build_record_strips_overlay_and_image() -> None:
    trace_dict = {
        "answer": "yes",
        "overlay_b64": "PNGBASE64==",
        "steps": [],
        "attention_map": {"width": 1, "height": 1, "data": [[0.0]]},
        "backend": "mock",
        "latency_ms": 1.0,
    }
    r = build_record(trace_dict, image_b64="ABCD", question="Q?")
    assert r["question"] == "Q?"
    assert r["image_sha256"] == hash_image("ABCD")
    assert "ABCD" not in json.dumps(r)
    assert "overlay_b64" not in r["trace"]
    assert "PNGBASE64==" not in json.dumps(r)
    # Trace fields preserved
    assert r["trace"]["answer"] == "yes"
    assert r["trace"]["backend"] == "mock"


def test_build_record_no_image_yields_null_hash() -> None:
    r = build_record({"answer": "x"}, image_b64=None, question="Q")
    assert r["image_sha256"] is None


def test_record_includes_iso_timestamp() -> None:
    r = build_record({"answer": "x"}, image_b64="A", question="Q")
    # Round-trips through fromisoformat without raising.
    from datetime import datetime

    datetime.fromisoformat(r["ts"])


# ---------------------------------------------------------------------------
# Env gating
# ---------------------------------------------------------------------------


def test_is_recording_enabled_truthy(monkeypatch) -> None:
    for v in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("MIRU_RECORD", v)
        assert is_recording_enabled() is True


def test_is_recording_enabled_falsy(monkeypatch) -> None:
    monkeypatch.delenv("MIRU_RECORD", raising=False)
    assert is_recording_enabled() is False
    monkeypatch.setenv("MIRU_RECORD", "0")
    assert is_recording_enabled() is False
    monkeypatch.setenv("MIRU_RECORD", "no")
    assert is_recording_enabled() is False


def test_maybe_record_noop_when_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MIRU_RECORD", raising=False)
    monkeypatch.setenv("MIRU_RECORD_PATH", str(tmp_path))
    reset_recorder()
    maybe_record({"answer": "x"}, image_b64="A", question="Q")
    # Nothing should have been written
    assert not list(tmp_path.glob("*.jsonl"))


# ---------------------------------------------------------------------------
# TraceRecorder lifecycle & file output
# ---------------------------------------------------------------------------


def test_recorder_writes_jsonl_lines(tmp_path) -> None:
    rec = TraceRecorder(str(tmp_path), batch_size=4, flush_interval=0.1)
    rec.start()
    try:
        for i in range(3):
            rec.enqueue({"i": i, "ts": "now"})
        rec.flush()
    finally:
        rec.stop()
    files = sorted(tmp_path.glob("traces-*.jsonl"))
    assert files
    lines = [line for f in files for line in f.read_text().splitlines()]
    assert len(lines) == 3
    assert sorted(json.loads(line)["i"] for line in lines) == [0, 1, 2]


def test_recorder_flush_returns_count(tmp_path) -> None:
    rec = TraceRecorder(str(tmp_path))
    rec.enqueue({"a": 1})
    rec.enqueue({"a": 2})
    written = rec.flush()
    assert written == 2
    assert rec.flush() == 0  # idempotent


def test_recorder_stop_drains_queue(tmp_path) -> None:
    rec = TraceRecorder(str(tmp_path), flush_interval=0.05)
    rec.start()
    for i in range(5):
        rec.enqueue({"i": i})
    rec.stop(timeout=2.0)
    files = list(tmp_path.glob("traces-*.jsonl"))
    assert files
    total = sum(len(f.read_text().splitlines()) for f in files)
    assert total == 5


def test_recorder_batches_above_batch_size(tmp_path) -> None:
    rec = TraceRecorder(str(tmp_path), batch_size=3)
    for i in range(7):
        rec.enqueue({"i": i})
    rec.flush()
    files = sorted(tmp_path.glob("traces-*.jsonl"))
    # 7 records / batch_size 3 => 3 files (3, 3, 1)
    assert len(files) == 3
    sizes = [len(f.read_text().splitlines()) for f in files]
    assert sizes == [3, 3, 1]
    all_lines = [line for f in files for line in f.read_text().splitlines()]
    assert [json.loads(l)["i"] for l in all_lines] == list(range(7))


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def test_get_recorder_is_singleton(recording_dir) -> None:
    r1 = rec_mod.get_recorder()
    r2 = rec_mod.get_recorder()
    assert r1 is r2


def test_reset_recorder_creates_fresh_instance(recording_dir) -> None:
    r1 = rec_mod.get_recorder()
    reset_recorder()
    r2 = rec_mod.get_recorder()
    assert r1 is not r2


# ---------------------------------------------------------------------------
# fsspec backend (memory://)
# ---------------------------------------------------------------------------


def test_recorder_writes_to_fsspec_uri() -> None:
    fsspec = pytest.importorskip("fsspec")
    # Use the in-memory filesystem so no real I/O occurs.
    target = "memory:///miru-recorder-test"
    fs = fsspec.filesystem("memory")
    # Wipe any leftover state from prior tests in same process.
    if fs.exists(target):
        fs.rm(target, recursive=True)

    rec = TraceRecorder(target, batch_size=2, flush_interval=0.1)
    try:
        rec.enqueue({"hello": "memory"})
        rec.flush()
        listing = fs.ls(target)
        names = [e["name"] if isinstance(e, dict) else e for e in listing]
        jsonl_files = [n for n in names if "traces-" in n and n.endswith(".jsonl")]
        assert jsonl_files, f"expected JSONL file under {target}, saw {names}"
        with fsspec.open(f"memory://{jsonl_files[0]}", "r", encoding="utf-8") as f:
            content = f.read()
        assert json.loads(content.strip())["hello"] == "memory"
    finally:
        rec.stop()
        if fs.exists(target):
            fs.rm(target, recursive=True)


# ---------------------------------------------------------------------------
# /analyze hook
# ---------------------------------------------------------------------------


def test_analyze_records_when_enabled(client: TestClient, mock_image_b64: str, recording_dir) -> None:
    payload = {"image_b64": mock_image_b64, "question": "recorded?", "backend": "mock"}
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200

    rec_mod.get_recorder().flush()

    files = list(recording_dir.glob("traces-*.jsonl"))
    assert files, f"expected JSONL output in {recording_dir}, saw {list(recording_dir.iterdir()) if recording_dir.exists() else 'no dir'}"
    lines = files[0].read_text().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert rec["question"] == "recorded?"
    assert rec["image_sha256"] == hash_image(mock_image_b64)
    assert "overlay_b64" not in rec["trace"]
    # Privacy: the original image_b64 must never appear anywhere in the record.
    assert mock_image_b64 not in lines[-1]


def test_analyze_does_not_record_when_disabled(
    client: TestClient, mock_image_b64: str, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("MIRU_RECORD", raising=False)
    monkeypatch.setenv("MIRU_RECORD_PATH", str(tmp_path / "should_not_exist"))
    reset_recorder()
    try:
        resp = client.post(
            "/analyze",
            json={"image_b64": mock_image_b64, "question": "q", "backend": "mock"},
        )
        assert resp.status_code == 200
        assert not (tmp_path / "should_not_exist").exists()
    finally:
        reset_recorder()


# ---------------------------------------------------------------------------
# /analyze/stream hook
# ---------------------------------------------------------------------------


def test_analyze_stream_records_when_enabled(
    client: TestClient, mock_image_b64: str, recording_dir
) -> None:
    # Ensure registry has defaults (test_registry tears it down)
    from miru.models import registry

    registry.register_defaults()

    resp = client.post(
        "/analyze/stream",
        json={"image_b64": mock_image_b64, "question": "stream-record", "backend": "mock"},
    )
    assert resp.status_code == 200

    rec_mod.get_recorder().flush()

    files = list(recording_dir.glob("traces-*.jsonl"))
    assert files
    lines = files[0].read_text().splitlines()
    questions = [json.loads(l)["question"] for l in lines]
    assert "stream-record" in questions
