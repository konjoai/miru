"""``miru record …`` subcommand implementations."""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

from miru.recorder import (
    DEFAULT_PATH,
    RECORD_PATH_ENV,
    _is_uri,
    _list_files,
    _read_lines,
)


def _resolve_path(explicit: Optional[str]) -> str:
    return explicit or os.environ.get(RECORD_PATH_ENV) or DEFAULT_PATH


def _file_size(path: str) -> int:
    """File size in bytes — local stat, or fsspec metadata for URIs."""
    if _is_uri(path):
        import fsspec  # type: ignore

        fs, _, paths = fsspec.get_fs_token_paths(path)
        info = fs.info(paths[0])
        return int(info.get("size", 0))
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def _count_lines(path: str) -> int:
    return sum(1 for _ in _read_lines(path))


def run_list(explicit_path: Optional[str], stream=None) -> int:
    """Print a one-line-per-file inventory of the recording directory.

    Format::

        <records>\\t<bytes>\\t<path>

    Returns process exit code (0 on success).  When the directory does not
    exist or is empty, prints ``no recorded traces`` and still exits 0.
    """
    out = stream or sys.stdout
    directory = _resolve_path(explicit_path)
    files = _list_files(directory)
    if not files:
        out.write(f"no recorded traces in {directory}\n")
        return 0
    for path in files:
        records = _count_lines(path)
        size = _file_size(path)
        out.write(f"{records}\t{size}\t{path}\n")
    return 0


def _iter_records(directory: str) -> Iterator[dict]:
    for path in _list_files(directory):
        for line in _read_lines(path):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Corrupt line — skip rather than fail the whole export.
                continue


_CSV_FIELDS = (
    "ts",
    "question",
    "image_sha256",
    "answer",
    "backend",
    "latency_ms",
    "n_steps",
)


def _flatten_for_csv(record: dict) -> dict[str, object]:
    trace = record.get("trace") or {}
    return {
        "ts": record.get("ts", ""),
        "question": record.get("question", ""),
        "image_sha256": record.get("image_sha256", ""),
        "answer": trace.get("answer", ""),
        "backend": trace.get("backend", ""),
        "latency_ms": trace.get("latency_ms", ""),
        "n_steps": len(trace.get("steps", [])),
    }


def run_export(
    explicit_path: Optional[str],
    out_path: str,
    fmt: str,
    stream=None,
) -> int:
    """Concatenate every recorded JSONL line into *out_path*.

    JSONL: byte-faithful re-serialization (compact separators).
    CSV:   flattened summary with the columns in :data:`_CSV_FIELDS`.

    Returns 0 on success, 2 on invalid format.
    """
    out = stream or sys.stdout
    directory = _resolve_path(explicit_path)
    if fmt == "jsonl":
        count = _export_jsonl(directory, out_path)
    elif fmt == "csv":
        count = _export_csv(directory, out_path)
    else:  # pragma: no cover — argparse rejects invalid values
        out.write(f"unknown format: {fmt}\n")
        return 2
    out.write(f"wrote {count} records to {out_path}\n")
    return 0


def _export_jsonl(directory: str, out_path: str) -> int:
    count = 0
    # Truncate the destination first by opening for write, then append below.
    if _is_uri(out_path):
        import fsspec  # type: ignore

        with fsspec.open(out_path, mode="w", encoding="utf-8") as f:
            for rec in _iter_records(directory):
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                count += 1
        return count
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for rec in _iter_records(directory):
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            count += 1
    return count


def _export_csv(directory: str, out_path: str) -> int:
    count = 0
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(_CSV_FIELDS))
    writer.writeheader()
    for rec in _iter_records(directory):
        writer.writerow(_flatten_for_csv(rec))
        count += 1
    if _is_uri(out_path):
        import fsspec  # type: ignore

        with fsspec.open(out_path, mode="w", encoding="utf-8", newline="") as f:
            f.write(buf.getvalue())
        return count
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
    return count


__all__ = ["run_list", "run_export"]
