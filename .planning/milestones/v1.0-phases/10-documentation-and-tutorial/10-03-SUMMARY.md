---
phase: 10
plan: 03
subsystem: documentation
tags: [tutorial, docs, DOC-03, BaseSSLModule, dispatcher]
dependency_graph:
  requires: [10-01]
  provides: [docs/tutorial/01_add_a_new_method.md, DOC-03-section-a]
  affects: [10-06]
tech_stack:
  added: []
  patterns: [Markdown tutorial, real-code snippets, 6-step walkthrough]
key_files:
  created:
    - docs/tutorial/01_add_a_new_method.md
  modified: []
decisions:
  - "Tutorial written as Markdown (not Jupyter) per RESEARCH.md recommendation — no Jupyter dependency, GitHub-renderable"
  - "MyToyContrastiveModule example exists only as text in Markdown — not added to methods/ (tutorial fixture only)"
  - "InfoNCELoss used in symmetric mode (same path as SimCLR) for the minimal example"
metrics:
  duration_seconds: 180
  completed_date: "2026-05-03"
  tasks_completed: 1
  files_created: 1
---

# Phase 10 Plan 03: Tutorial Section (a) — Add a New Method Summary

Wrote `docs/tutorial/01_add_a_new_method.md`: 6-step walkthrough showing how to subclass BaseSSLModule, register via method_dispatcher, write a YAML config, and train with python train.py — using real repo imports throughout.

## What Was Built

`docs/tutorial/01_add_a_new_method.md` (8980 bytes, 257 lines) — standalone tutorial section for DOC-03 requirement (a). Covers:

1. The interface BaseSSLModule expects (abstract methods, shared infrastructure)
2. Step 1: Create the package directory structure
3. Step 2: Implement MyToyContrastiveModule (full working code block with real imports)
4. Step 3: Register with the dispatcher (register_method + methods/__init__.py line)
5. Step 4: Write a YAML config (with TrainConfig validation rules explained)
6. Step 5: Run training with python train.py
7. Step 6: Add a smoke test (full pytest-compatible test template)
8. "Where to go next" section pointing to EMA/queue/multi-crop extensions

## Acceptance Criteria Verification

All 16 acceptance criteria passed:

- File exists at `docs/tutorial/01_add_a_new_method.md`
- Contains `# Tutorial Section (a): How to Add a New Method` heading
- Has all 6 step headings (Step 1 through Step 6)
- Has `## The interface: what BaseSSLModule expects` section
- Contains `class MyToyContrastiveModule(BaseSSLModule)` class definition
- Contains `register_method("my_toy_contrastive", MyToyContrastiveModule)` call
- Contains `from core.base import BaseSSLModule`
- Contains `from core.backbone import build_backbone`
- Contains `from core.projection import ProjectionHead`
- Contains `from core.losses import InfoNCELoss`
- Contains `method: my_toy_contrastive` YAML line
- Contains `python train.py --config configs/my_toy_contrastive_resnet18.yaml`
- File size 8980 bytes (>5KB requirement)
- No regressions (docs-only change — no Python code modified)

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write tutorial section (a) | 12bbbb9 | docs/tutorial/01_add_a_new_method.md |

## Deviations from Plan

None — plan executed exactly as written. The file content was taken verbatim from the plan's "BEGIN FILE CONTENT / END FILE CONTENT" block.

## Known Stubs

None. The tutorial is complete prose with working code examples. No placeholder text or TODO markers.

## Threat Flags

No new threat surface introduced. This plan creates a Markdown documentation file only — no network endpoints, auth paths, file system writes, or schema changes.

## Self-Check: PASSED

- [x] `docs/tutorial/01_add_a_new_method.md` — FOUND (8980 bytes)
- [x] Commit 12bbbb9 — FOUND in git log
