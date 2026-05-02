# 🐍 Miru

![Language](https://img.shields.io/badge/language-python-yellow) ![API](https://img.shields.io/badge/framework-fastapi-green) ![License](https://img.shields.io/badge/license-busl--1.1-green)

> 👁️ Multimodal reasoning tracer — see what models see, and why they decide.

---

## 👁️ Meaning

**Miru (見る)** — *to see, to observe.*

Not just outputs — but perception and reasoning.

---

## 🚀 What it is

Miru is a **multimodal explainability engine**:

* Input: image or document + question
* Output:

  * answer
  * reasoning trace
  * attention visualization

---

## ❗ The problem

Multimodal models are black boxes:

* No visibility into reasoning
* No auditability
* No explainability

Critical issue for:

* compliance
* medical
* enterprise AI

---

## 🧠 What you learn

* Vision-language models (VLMs)
* Cross-attention mechanisms
* Saliency & interpretability
* Multimodal reasoning

---

## ⚙️ Stack

* 🐍 Python (FastAPI backend)
* 🎨 Visualization layer (attention maps, overlays)

---

## 🚀 Quick Start

```bash
uvicorn miru.main:app --reload
```

### Endpoints

* `GET  /health` — service status + registered backends
* `POST /analyze` — synchronous reasoning trace; `?overlay=true` returns base64 PNG attention overlay
* `POST /analyze/stream` — Server-Sent Events. Emits `step` events as each reasoning step lands, then a final `trace` event (schema-equivalent to `/analyze`), then `done`. Supports `?overlay=true` and `?timeout_seconds=<1..300>` (default 30s).

### Recording traces

Set `MIRU_RECORD=1` (and optionally `MIRU_RECORD_PATH=<dir-or-uri>`) to persist every trace as JSONL. Raw image bytes are never stored — only the SHA-256 of the source `image_b64` payload, alongside the question, an ISO-8601 UTC timestamp, and the trace itself (with the overlay stripped). Local paths and any `fsspec`-supported URI (`s3://…`, `gs://…`, `memory://…`) are accepted; install `fsspec` via `pip install miru[storage]` for cloud backends.

```bash
miru record list                    # inventory of recorded files
miru record export --out all.jsonl  # concatenate everything into one JSONL
miru record export --out summary.csv --format csv
```

---

## 🎯 Vision

> Make AI reasoning visible.
