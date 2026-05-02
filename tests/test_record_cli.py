"""Tests for the ``miru record`` CLI."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from miru.cli import build_parser, main
from miru.cli.record import run_export, run_list


def _seed(directory: Path, records: list[dict], filename: str = "traces-20260501T000000-000000.jsonl") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _sample_record(question: str = "Q", answer: str = "A") -> dict:
    return {
        "ts": "2026-05-01T00:00:00+00:00",
        "question": question,
        "image_sha256": "0" * 64,
        "trace": {
            "answer": answer,
            "backend": "mock",
            "latency_ms": 1.5,
            "steps": [{"step": 1, "description": "s", "confidence": 0.9}],
            "attention_map": {"width": 1, "height": 1, "data": [[0.0]]},
        },
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_accepts_record_list_and_export() -> None:
    parser = build_parser()
    a = parser.parse_args(["record", "list", "--path", "/tmp/x"])
    assert (a.cmd, a.record_cmd, a.path) == ("record", "list", "/tmp/x")
    a = parser.parse_args(["record", "export", "--out", "/tmp/o.jsonl"])
    assert (a.cmd, a.record_cmd, a.format, a.out) == ("record", "export", "jsonl", "/tmp/o.jsonl")


# ---------------------------------------------------------------------------
# `record list`
# ---------------------------------------------------------------------------


def test_list_empty_directory(tmp_path) -> None:
    buf = io.StringIO()
    code = run_list(str(tmp_path / "empty"), stream=buf)
    assert code == 0
    assert "no recorded traces" in buf.getvalue()


def test_list_reports_records_and_size(tmp_path) -> None:
    _seed(tmp_path, [_sample_record("a"), _sample_record("b"), _sample_record("c")])
    buf = io.StringIO()
    code = run_list(str(tmp_path), stream=buf)
    assert code == 0
    line = buf.getvalue().strip()
    parts = line.split("\t")
    assert int(parts[0]) == 3
    assert int(parts[1]) > 0
    assert "traces-" in parts[2] and parts[2].endswith(".jsonl")


def test_list_main_entrypoint(tmp_path, capsys) -> None:
    _seed(tmp_path, [_sample_record()])
    code = main(["record", "list", "--path", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "traces-" in out and ".jsonl" in out


# ---------------------------------------------------------------------------
# `record export jsonl`
# ---------------------------------------------------------------------------


def test_export_jsonl_concatenates(tmp_path) -> None:
    _seed(tmp_path, [_sample_record("q1"), _sample_record("q2")], filename="traces-20260501T000000-000000.jsonl")
    _seed(tmp_path, [_sample_record("q3")], filename="traces-20260502T000000-000000.jsonl")
    out_path = tmp_path / "out.jsonl"
    buf = io.StringIO()
    code = run_export(str(tmp_path), str(out_path), "jsonl", stream=buf)
    assert code == 0
    lines = out_path.read_text().splitlines()
    assert len(lines) == 3
    questions = [json.loads(l)["question"] for l in lines]
    assert sorted(questions) == ["q1", "q2", "q3"]
    assert "wrote 3 records" in buf.getvalue()


def test_export_jsonl_skips_corrupt_lines(tmp_path) -> None:
    path = tmp_path / "traces-20260501T000000-000000.jsonl"
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(_sample_record("good")) + "\n")
        f.write("not-json\n")
        f.write(json.dumps(_sample_record("also-good")) + "\n")
    out = tmp_path / "out.jsonl"
    code = run_export(str(tmp_path), str(out), "jsonl")
    assert code == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# `record export csv`
# ---------------------------------------------------------------------------


def test_export_csv_flattens_fields(tmp_path) -> None:
    _seed(tmp_path, [_sample_record("hello", "world"), _sample_record("foo", "bar")])
    out = tmp_path / "out.csv"
    code = run_export(str(tmp_path), str(out), "csv")
    assert code == 0
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert len(rows) == 2
    assert {r["question"] for r in rows} == {"hello", "foo"}
    assert {r["answer"] for r in rows} == {"world", "bar"}
    assert all(r["backend"] == "mock" for r in rows)
    assert all(int(r["n_steps"]) == 1 for r in rows)


def test_export_main_entrypoint_csv(tmp_path) -> None:
    _seed(tmp_path, [_sample_record()])
    out = tmp_path / "out.csv"
    code = main(["record", "export", "--path", str(tmp_path), "--out", str(out), "--format", "csv"])
    assert code == 0
    assert out.exists()
    assert "question" in out.read_text().splitlines()[0]
