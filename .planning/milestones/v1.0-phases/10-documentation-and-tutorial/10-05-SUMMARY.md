---
phase: 10
plan: 05
subsystem: docs
tags: [documentation, tutorial, comparison, eval-suite, simclr, moco]
dependency_graph:
  requires: [10-01, 10-03, 10-04]
  provides: [docs/tutorial/03_comparing_methods.md]
  affects: [10-06]
tech_stack:
  added: []
  patterns: [positional-config-then-ckpt CLI pattern, shell-variable checkpoint paths]
key_files:
  created:
    - docs/tutorial/03_comparing_methods.md
  modified: []
decisions:
  - Use $SIMCLR_CKPT / $MOCO_CKPT shell variables for checkpoint paths — reader substitutes their own paths, reduces command verbosity across 5 steps
  - Use 0.<XYZ> placeholders for all accuracy numbers — real values depend on the user's training run (T-10-10 threat mitigation)
  - Include UMAP and CAM commands under "Optional" step — they exist in the eval suite and are valuable, but not required for the basic comparison
metrics:
  duration_seconds: 180
  completed_date: "2026-05-03"
  tasks_completed: 1
  files_created: 1
  files_modified: 0
---

# Phase 10 Plan 05: Comparing Two Methods Tutorial Summary

Tutorial section (c) of the DOC-03 walkthrough — a 7-step guide comparing SimCLR v1 vs MoCo v2 using the full evaluation suite, with fair-comparison gotchas and interpretation rules of thumb.

## What Was Built

`docs/tutorial/03_comparing_methods.md` (190 lines, 7.6 KB) — a worked example tutorial section that walks a reader through:

1. Training both methods with matched hyperparameters (backbone, dataset, budget)
2. Running linear probe on both checkpoints
3. Running k-NN evaluation on both checkpoints
4. Running t-SNE perplexity sweep on both checkpoints
5. Optional UMAP and CAM for additional confirmation
6. Building a Markdown comparison table with `0.<XYZ>` placeholders
7. Interpreting differences (rules of thumb for signal vs noise)

The section also covers what NOT to compare (different backbones, different epoch counts, different effective batch sizes) and lists 6 other comparison pairings for readers who want to explore further.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write docs/tutorial/03_comparing_methods.md | 74e857d | docs/tutorial/03_comparing_methods.md |

## Verification Results

All acceptance criteria passed:

- File exists at `docs/tutorial/03_comparing_methods.md`
- Heading `# Tutorial Section (c): Comparing Two Methods` present
- 7 `## Step` headings (meets >= 6 requirement)
- References both `configs/simclr_v1_resnet18.yaml` and `configs/moco_v2_resnet18.yaml`
- 2x `python eval/linear_probe.py` commands (one per checkpoint)
- 2x `python eval/tsne_vis.py` commands (one per checkpoint)
- 2x `python eval/umap_vis.py` commands (one per checkpoint)
- 2x `python eval/cam_vis.py` commands (one per checkpoint)
- Comparison table with `Linear probe top-1` and `k-NN top-1` rows
- `Other comparison pairings` section present
- `What is NOT comparable` subsection present
- `rules of thumb` section present
- File size: 7609 bytes (>= 5000)

## Deviations from Plan

None — plan executed exactly as written. The `docs/tutorial/` directory was created as part of this plan (the plan notes to create it if Plan 10-03 has not done so yet).

Note: `tests/test_collapse_monitoring.py::test_corr_diag_mean_in_valid_range` was observed failing during verification with a floating-point precision issue (`1.0000028610229492 > 1.0`). This is a pre-existing flaky test in the Barlow Twins collapse monitoring — it passes on `main` and fails intermittently due to floating-point bounds in the test assertion. It is unrelated to this plan's doc-only changes and is logged here for tracking.

## Threat Model Compliance

T-10-10 (Information Disclosure — hard-coded accuracy numbers): Mitigated. All metric values use `0.<XYZ>` placeholders throughout the document. The table explicitly notes "exact values depend on your training run."

## Known Stubs

None. The tutorial is a complete, executable guide. All referenced scripts (`eval/linear_probe.py`, `eval/tsne_vis.py`, `eval/umap_vis.py`, `eval/cam_vis.py`) exist and are production-ready from Phase 9.

## Threat Flags

None. This plan creates a documentation-only file with no network endpoints, auth paths, file access patterns, or schema changes.

## Self-Check: PASSED

- FOUND: docs/tutorial/03_comparing_methods.md
- FOUND: commit 74e857d (feat(10-05): write tutorial section (c) — comparing two methods)
