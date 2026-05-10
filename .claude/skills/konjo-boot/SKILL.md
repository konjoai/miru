---
name: konjo-boot
description: Boot a Konjo session for miru. Produces a Session Brief, runs Discovery, identifies the next sprint. Use at the start of any work session or when invoked with /konjo.
user-invocable: true
---
# Konjo Session Boot — miru

## Step 1 — Read
Read in order: CLAUDE.md, README.md, CHANGELOG.md, PLAN.md, docs/ (if it exists).

## Step 2 — Session Brief
```
REPO         miru — multimodal reasoning tracer (attention maps, VLM interpretability, training data)
LAST SHIPPED [most recent meaningful change from CHANGELOG.md]
OPEN WORK    [stated next steps from PLAN.md]
BLOCKERS     [failing tests, broken modules, open issues]
HEALTH       [Green / Yellow / Red — one line]
```

## Step 3 — Discovery
Search arXiv (VLM attention, interpretability, multimodal reasoning), GitHub (VLM frameworks), HuggingFace (CLIP, LLaVA updates).

## Step 4 — Identify Work
Load PLAN.md, validate against codebase, flag drift.

## Invocation Keywords
- `konjo` / `konjo miru` / `miru konjo` / `read KONJO_PROMPT.md and begin`
