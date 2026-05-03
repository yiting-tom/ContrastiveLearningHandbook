---
phase: 10-documentation-and-tutorial
plan: "04"
subsystem: docs
tags: [tutorial, simclr, cifar10, linear-probe, umap, tensorboard, evaluation]

# Dependency graph
requires:
  - phase: 10-01
    provides: train.py CLI entry point with --config / --data-dir / --ckpt-path flags
  - phase: 09-evaluation-suite
    provides: eval/*.py scripts with positional config + --ckpt signature

provides:
  - "docs/tutorial/02_running_an_experiment.md — DOC-03 section (b) annotated walkthrough"

affects: [10-06-assembly, readers following end-to-end tutorial]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "9-step walkthrough covering full experiment lifecycle: config -> data prep -> train -> checkpoint -> eval -> visualization"
    - "All CLI commands use verified positional-config + --ckpt signatures from eval/*.py"
    - "CIFAR-10 ImageFolder conversion as copy-pasteable Python heredoc"

key-files:
  created:
    - docs/tutorial/02_running_an_experiment.md
  modified: []

key-decisions:
  - "Tutorial uses real CLI signatures: eval scripts take config as positional arg, not --config flag"
  - "Accuracy ranges (0.85-0.92 linear probe, 0.80-0.88 k-NN) cited as literature-typical sanity-check guidance, not hard targets"
  - "CIFAR-10 conversion included as inline heredoc for copy-paste convenience, not parameterized"

patterns-established:
  - "Section (b) pattern: numbered steps with expected outputs + troubleshooting table"

requirements-completed: [DOC-03]

# Metrics
duration: 8min
completed: 2026-05-03
---

# Phase 10 Plan 04: Running an Experiment End-to-End Summary

**9-step SimCLR-on-CIFAR-10 tutorial with verified CLI commands, TensorBoard scalar keys, accuracy ranges, and a sanity-check troubleshooting table**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-05-03T08:00:00Z
- **Completed:** 2026-05-03T08:08:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `docs/tutorial/02_running_an_experiment.md` (8.9 KB, 236 lines) — DOC-03 section (b)
- Covers full experiment lifecycle: config selection -> ImageFolder data prep -> training -> checkpoint location -> linear probe -> UMAP/t-SNE -> k-NN -> fine-tune -> CAM visualization
- All 9 eval commands use real CLI signatures (`<config.yaml> --ckpt <path>`) verified against `eval/*.py` scripts
- Documents the 3 key TensorBoard scalars (`train/loss`, `train/lr`, `eval/knn_acc`) and expected accuracy ranges for CIFAR-10
- Includes copy-pasteable CIFAR-10 ImageFolder conversion snippet and 5-row sanity-check troubleshooting table

## Task Commits

1. **Task 1: Write docs/tutorial/02_running_an_experiment.md** - `0bccbd3` (feat)

**Plan metadata:** (SUMMARY commit follows)

## Files Created/Modified

- `docs/tutorial/02_running_an_experiment.md` — Tutorial section (b): 9-step end-to-end experiment walkthrough for SimCLR/CIFAR-10

## Decisions Made

- All `python eval/<script>.py` invocations use the verified `<config.yaml> --ckpt <path>` positional signature (not `--config configs/...`)
- Accuracy ranges documented as literature-typical guidance, not hard targets
- CIFAR-10 ImageFolder conversion written as inline heredoc (copy-pasteable without modification)

## Deviations from Plan

None — plan executed exactly as written. The pre-existing `tests/test_collapse_monitoring.py::test_corr_diag_mean_in_valid_range` failure exists in the base commit and is unrelated to this documentation-only plan.

## Known Stubs

None — the document provides real commands, real scalar keys, and real accuracy ranges grounded in the existing codebase.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Document is read-only for users.

## Issues Encountered

- Pre-existing test failure in `tests/test_collapse_monitoring.py::test_corr_diag_mean_in_valid_range` was present before this plan and is unrelated to documentation changes. All other tests pass (exit 0).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `docs/tutorial/02_running_an_experiment.md` is ready for assembly by Plan 10-06
- Plan 10-05 (section (c) — comparing methods) can reference this document as the base workflow

## Self-Check

- `docs/tutorial/02_running_an_experiment.md` exists: FOUND
- Commit `0bccbd3` exists: FOUND
- File size: 8931 bytes (> 5000 threshold)
- Step headings: 9 (acceptance criterion: >= 8)

---
*Phase: 10-documentation-and-tutorial*
*Completed: 2026-05-03*
