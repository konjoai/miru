"""Miru demo — exercises every public surface of the library.

Run::

    python -m demo.demo

or::

    python demo/demo.py

The demo uses the FastAPI test client and the deterministic mock backend so
no real VLM weights are required.  Output is rendered with ``rich`` for a
high-bandwidth visual presentation suitable for a demo-day audience.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Allow ``python demo/demo.py`` from a fresh checkout without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from fastapi.testclient import TestClient
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.json import JSON
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from miru import __version__
from miru.cli.record import run_export, run_list
from miru.main import app
from miru.models import registry
from miru.recorder import (
    TraceRecorder,
    build_record,
    hash_image,
    reset_recorder,
)


console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BANNER = r"""
   __  ____
  /  |/  (_)______ __
 / /|_/ / / __/ // /
/_/  /_/_/_/  \_,_/
"""


def _section(title: str, subtitle: str | None = None) -> None:
    """Render a chapter divider."""
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/]", style="cyan"))
    if subtitle:
        console.print(f"[dim]{subtitle}[/]")
    console.print()


def _make_synthetic_image_b64(size: int = 8) -> str:
    """Produce a deterministic synthetic image as base64 raw bytes."""
    rng = np.random.default_rng(seed=1729)
    arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    return base64.b64encode(arr.tobytes()).decode()


def _ensure_registry() -> None:
    registry.register_defaults()


# ---------------------------------------------------------------------------
# Section 1 — Library + routes
# ---------------------------------------------------------------------------


def section_intro() -> None:
    title = Text(_BANNER, style="bold magenta", justify="left")
    blurb = Text.from_markup(
        f"[bold]Miru v{__version__}[/]  —  multimodal reasoning tracer\n"
        "[dim]see what models see, and why they decide.[/]"
    )
    console.print(Panel(Group(title, blurb), border_style="magenta", padding=(1, 4)))


def section_routes() -> None:
    _section(
        "1 / FastAPI surface",
        "Every public route, materialised live from the running app.",
    )
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, header_style="bold cyan")
    table.add_column("Method", style="green")
    table.add_column("Path", style="bold")
    table.add_column("Summary", style="dim")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        method_str = ", ".join(sorted(m for m in methods if m != "HEAD"))
        summary = (route.endpoint.__doc__ or "").strip().splitlines()[0] if route.endpoint.__doc__ else ""
        table.add_row(method_str, path, summary or "—")
    console.print(table)


# ---------------------------------------------------------------------------
# Section 2 — POST /analyze
# ---------------------------------------------------------------------------


def section_analyze(client: TestClient, image_b64: str) -> dict:
    _section(
        "2 / POST /analyze",
        "Single-shot synchronous reasoning trace from the mock backend.",
    )
    payload = {
        "image_b64": image_b64,
        "question": "What is the dominant color in this image?",
        "backend": "mock",
    }

    console.print(
        Panel(
            JSON.from_data({k: (v if k != "image_b64" else f"<{len(v)}-char base64>") for k, v in payload.items()}),
            title="[bold]request[/]",
            border_style="blue",
        )
    )

    t0 = time.perf_counter()
    resp = client.post("/analyze", json=payload)
    dt = (time.perf_counter() - t0) * 1000
    data = resp.json()

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("status", f"[green]{resp.status_code}[/]")
    summary.add_row("backend", f"[bold]{data['backend']}[/]")
    summary.add_row("latency (server)", f"[yellow]{data['latency_ms']:.2f} ms[/]")
    summary.add_row("latency (round-trip)", f"[yellow]{dt:.2f} ms[/]")
    summary.add_row("answer", f"[bold green]{data['answer']!r}[/]")
    console.print(Panel(summary, title="[bold]response summary[/]", border_style="green"))

    # Reasoning steps with confidence bars.
    steps_tbl = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    steps_tbl.add_column("#", justify="right", width=3)
    steps_tbl.add_column("description")
    steps_tbl.add_column("confidence", justify="right", width=14)
    steps_tbl.add_column("", width=22)
    for s in data["steps"]:
        bar_len = int(s["confidence"] * 20)
        bar = "[green]" + "█" * bar_len + "[/]" + "[dim]" + "░" * (20 - bar_len) + "[/]"
        steps_tbl.add_row(
            str(s["step"]),
            s["description"],
            f"{s['confidence']:.3f}",
            bar,
        )
    console.print(Panel(steps_tbl, title="[bold]reasoning trace[/]", border_style="cyan"))

    # Attention map preview — render the 16x16 grid as a heatmap.
    attn = np.asarray(data["attention_map"]["data"], dtype=float)
    console.print(Panel(_render_heatmap(attn), title="[bold]attention map (16×16)[/]", border_style="magenta"))

    return data


def _render_heatmap(arr: np.ndarray) -> Text:
    """Render a 2-D float array in [0, 1] as a unicode block heatmap."""
    ramp = " ░▒▓█"
    palette = ["grey23", "grey50", "cyan", "bright_cyan", "bright_magenta"]
    out = Text()
    for row in arr:
        for v in row:
            v = float(max(0.0, min(1.0, v)))
            idx = min(len(ramp) - 1, int(v * (len(ramp) - 1)))
            out.append(ramp[idx], style=palette[idx])
        out.append("\n")
    return out


# ---------------------------------------------------------------------------
# Section 3 — POST /analyze/stream
# ---------------------------------------------------------------------------


def _parse_sse(stream_bytes: bytes) -> list[tuple[str, dict]]:
    """Minimal SSE frame parser — duplicate of the test helper."""
    text = stream_bytes.decode("utf-8")
    events: list[tuple[str, dict]] = []
    for raw_frame in text.split("\n\n"):
        frame = raw_frame.strip("\n")
        if not frame:
            continue
        event = "message"
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((event, json.loads("\n".join(data_lines))))
    return events


def section_stream(client: TestClient, image_b64: str) -> None:
    _section(
        "3 / POST /analyze/stream",
        "Server-Sent Events: each reasoning step arrives as it lands, then a final trace.",
    )
    payload = {
        "image_b64": image_b64,
        "question": "Describe what is happening in this scene.",
        "backend": "mock",
    }

    console.print("[dim]→ POST /analyze/stream[/]")
    resp = client.post("/analyze/stream", json=payload)
    events = _parse_sse(resp.content)

    for kind, data in events:
        if kind == "step":
            bar_len = int(float(data["confidence"]) * 20)
            bar = "[green]" + "█" * bar_len + "[/]"
            line = (
                f"[cyan]event:[/] [bold]step[/]   "
                f"[dim]#{data['step']}[/]  "
                f"{data['description']}  "
                f"{bar}"
            )
            console.print(line)
            time.sleep(0.18)  # demo cadence — make the streaming visible
        elif kind == "trace":
            console.print(
                f"[cyan]event:[/] [bold yellow]trace[/]  "
                f"[dim]answer=[/][bold]{data['answer']!r}[/]  "
                f"[dim]latency=[/][yellow]{data['latency_ms']:.2f} ms[/]"
            )
        elif kind == "done":
            console.print("[cyan]event:[/] [bold green]done[/]  [dim]stream closed cleanly.[/]")
        else:
            console.print(f"[red]event: {kind}[/] {data}")


# ---------------------------------------------------------------------------
# Section 4 — Recorder + CLI
# ---------------------------------------------------------------------------


def section_recorder_via_api(client: TestClient, image_b64: str) -> Path:
    _section(
        "4 / MIRU_RECORD=1 — auto-persisted traces",
        "Set the env var, hit /analyze, watch a JSONL appear on disk.",
    )
    record_dir = Path(tempfile.mkdtemp(prefix="miru-demo-record-"))
    os.environ["MIRU_RECORD"] = "1"
    os.environ["MIRU_RECORD_PATH"] = str(record_dir)
    reset_recorder()

    try:
        for question in (
            "What is the focus of this image?",
            "Are there any people present?",
            "Describe the lighting conditions.",
        ):
            client.post(
                "/analyze",
                json={"image_b64": image_b64, "question": question, "backend": "mock"},
            )

        # Force the background writer to flush so files are visible immediately.
        from miru.recorder import get_recorder

        get_recorder().flush()

        files = sorted(record_dir.glob("traces-*.jsonl"))
        files_tbl = Table(box=box.MINIMAL, header_style="bold cyan")
        files_tbl.add_column("file", style="bold")
        files_tbl.add_column("records", justify="right")
        files_tbl.add_column("bytes", justify="right")
        total = 0
        for f in files:
            n = len(f.read_text().splitlines())
            total += n
            files_tbl.add_row(f.name, str(n), str(f.stat().st_size))
        console.print(Panel(files_tbl, title=f"[bold]{record_dir}[/]", border_style="green"))
        console.print(f"[dim]total recorded:[/] [bold]{total}[/] traces")

        # Show one stored record (privacy-stripped) pretty-printed — redact the
        # attention_map data so the demo output stays readable.
        if files:
            raw = json.loads(files[0].read_text().splitlines()[0])
            attn = raw["trace"].get("attention_map", {})
            raw["trace"]["attention_map"] = {
                "width": attn.get("width"),
                "height": attn.get("height"),
                "data": f"<{attn.get('height', 0)}×{attn.get('width', 0)} float grid — elided for display>",
            }
            console.print(
                Panel(
                    JSON.from_data(raw),
                    title="[bold]first stored record (image_sha256 only, no raw bytes, no overlay)[/]",
                    border_style="green",
                )
            )

        # ``miru record list`` output captured in-process.
        console.print()
        console.print(Rule("[bold]$ miru record list[/]", style="dim"))
        buf = io.StringIO()
        run_list(str(record_dir), stream=buf)
        for line in buf.getvalue().splitlines():
            console.print(f"  [green]│[/] {line}")
    finally:
        reset_recorder()
        os.environ.pop("MIRU_RECORD", None)
        os.environ.pop("MIRU_RECORD_PATH", None)

    return record_dir


def section_recorder_direct(image_b64: str) -> None:
    _section(
        "5 / TraceRecorder direct API",
        "Skip the HTTP layer — hand the recorder synthetic traces directly, then export to CSV.",
    )
    record_dir = Path(tempfile.mkdtemp(prefix="miru-demo-direct-"))
    rec = TraceRecorder(str(record_dir), batch_size=8, flush_interval=0.2)
    rec.start()

    try:
        for i, (q, ans) in enumerate(
            [
                ("What objects are in this image?", "A natural outdoor scene."),
                ("What is the mood?", "Quiet, calm, well-lit."),
                ("Estimate the time of day.", "Afternoon, low-angle light."),
            ]
        ):
            trace_dict = {
                "answer": ans,
                "backend": "mock",
                "latency_ms": 0.7 + i * 0.15,
                "steps": [
                    {"step": 1, "description": "Identified salient regions.", "confidence": 0.95},
                    {"step": 2, "description": "Cross-referenced texture & color.", "confidence": 0.90},
                    {"step": 3, "description": "Synthesized final answer.", "confidence": 0.86},
                ],
                "attention_map": {"width": 1, "height": 1, "data": [[0.0]]},
                "overlay_b64": "PRETEND_PNG_BYTES",
            }
            record = build_record(trace_dict, image_b64=image_b64, question=q)
            rec.enqueue(record)
            console.print(
                f"[dim]enqueued[/]  "
                f"sha256={record['image_sha256'][:12]}…  "
                f"q=[bold]{q!r}[/]"
            )
        rec.flush()
        rec.stop()

        # Export to CSV using the public CLI helper.
        out_csv = record_dir / "summary.csv"
        run_export(str(record_dir), str(out_csv), "csv")
        console.print()
        console.print(Rule(f"[bold]{out_csv}[/]", style="dim"))
        console.print(
            Syntax(out_csv.read_text(), "csv", theme="ansi_dark", line_numbers=True)
        )
    finally:
        shutil.rmtree(record_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Privacy receipt
# ---------------------------------------------------------------------------


def section_privacy(image_b64: str) -> None:
    _section(
        "6 / Privacy receipt",
        "Verify what is — and is not — persisted.",
    )
    fake_trace = {"answer": "x", "overlay_b64": "RAW_PNG_OVERLAY_BYTES", "steps": []}
    record = build_record(fake_trace, image_b64=image_b64, question="Q?")
    serialised = json.dumps(record)

    receipts = Table.grid(padding=(0, 2))
    receipts.add_column(style="bold")
    receipts.add_column()
    receipts.add_row("[green]✓[/] image_sha256 stored", f"[dim]{record['image_sha256']}[/]")
    receipts.add_row("[green]✓[/] question stored", f"[dim]{record['question']}[/]")
    raw_present = "[red]LEAK[/]" if image_b64 in serialised else "[green]not present[/]"
    overlay_present = "[red]LEAK[/]" if "RAW_PNG_OVERLAY_BYTES" in serialised else "[green]stripped[/]"
    receipts.add_row("[green]✓[/] raw image_b64 in record", raw_present)
    receipts.add_row("[green]✓[/] overlay_b64 in record", overlay_present)
    console.print(Panel(receipts, border_style="green", title="[bold]privacy receipt[/]"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    section_intro()
    _ensure_registry()

    client = TestClient(app)
    image_b64 = _make_synthetic_image_b64(size=8)

    section_routes()
    section_analyze(client, image_b64)
    section_stream(client, image_b64)
    section_recorder_via_api(client, image_b64)
    section_recorder_direct(image_b64)
    section_privacy(image_b64)

    _section("done", "every public Miru surface exercised — no real VLM, no network.")
    console.print(
        Panel(
            Text.from_markup(
                "[bold green]✓[/] FastAPI routes\n"
                "[bold green]✓[/] /analyze (synchronous)\n"
                "[bold green]✓[/] /analyze/stream (SSE)\n"
                "[bold green]✓[/] MIRU_RECORD=1 auto-persistence\n"
                "[bold green]✓[/] miru record list / export\n"
                "[bold green]✓[/] TraceRecorder direct API\n"
                "[bold green]✓[/] privacy receipt"
            ),
            title="[bold magenta]demo complete[/]",
            border_style="magenta",
            padding=(1, 4),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
