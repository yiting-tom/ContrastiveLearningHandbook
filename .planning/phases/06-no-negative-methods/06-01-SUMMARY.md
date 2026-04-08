---
phase: 06-no-negative-methods
plan: "01"
subsystem: model
tags: [pytorch, ssl, byol, simsiam, predictor-head, batch-norm]

# Dependency graph
requires:
  - phase: 04-simclr-backbone
    provides: ProjectionHead pattern in core/projection.py
provides:
  - PredictorHead class in core/projection.py with standard (BYOL) and bottleneck (SimSiam) variants
affects:
  - 06-02-byol (imports PredictorHead for online branch predictor)
  - 06-03-simsiam (imports PredictorHead with bottleneck variant)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Predictor-on-online-branch: PredictorHead sits only on online branch, never on target/momentum branch"
    - "BN-no-ReLU-output: Both variants omit ReLU on output layer, applying BN only (standard SSL convention)"
    - "Bottleneck-by-default: bottleneck_dim defaults to 512 matching SimSiam paper (2048->512->2048)"

key-files:
  created:
    - tests/test_predictor_head.py
  modified:
    - core/projection.py

key-decisions:
  - "Implemented PredictorHead as a separate class (not subclass of ProjectionHead) for clarity — the bottleneck variant has different dim semantics (input_dim is both input and output, bottleneck_dim is the hidden dim)"
  - "bottleneck variant maps input_dim->bottleneck_dim->input_dim, making output_dim parameter unused for that variant — this matches SimSiam's fixed 2048->512->2048 architecture"
  - "ValueError raised eagerly in __init__ for invalid predictor_type so misconfiguration fails fast before any training"

patterns-established:
  - "PredictorHead import pattern: from core.projection import PredictorHead"
  - "TDD: RED commit (failing tests) before GREEN commit (implementation)"

requirements-completed:
  - ERA3-01
  - ERA3-02

# Metrics
duration: 15min
completed: 2026-04-08
---

# Phase 06 Plan 01: PredictorHead Summary

**PredictorHead MLP added to core/projection.py with BYOL standard variant (2-layer BN+ReLU->BN) and SimSiam bottleneck variant (2048->512->2048 with BN on all layers), verified by 7 unit tests**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-08T00:00:00Z
- **Completed:** 2026-04-08T00:15:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Implemented PredictorHead class in core/projection.py with standard and bottleneck variants
- Standard variant (BYOL): Linear->BN->ReLU->Linear->BN, no ReLU on output
- Bottleneck variant (SimSiam): 2048->512->2048, BN on ALL layers including output, no ReLU on output
- 7 unit tests passing covering forward shape, layer architecture, BN placement, ReLU absence on output, default bottleneck_dim=512, importability, and ValueError on invalid type

## Task Commits

Each task was committed atomically with TDD flow:

1. **Task 1 (RED): PredictorHead failing tests** - `20a67a5` (test)
2. **Task 1 (GREEN): PredictorHead implementation** - `cf7770e` (feat)

## Files Created/Modified
- `core/projection.py` - Added PredictorHead class (72 lines) after existing ProjectionHead; updated module docstring
- `tests/test_predictor_head.py` - 7 unit tests covering both variants (standard/BYOL and bottleneck/SimSiam)

## Decisions Made
- Implemented PredictorHead as a separate class rather than extending ProjectionHead — the bottleneck variant has distinct dimensional semantics (input_dim serves as both input and output dimension) that would complicate a shared interface
- `output_dim` parameter accepted but unused by the bottleneck variant to maintain a consistent constructor signature; bottleneck always produces output of size `input_dim`
- `ValueError` raised eagerly in `__init__` for unknown `predictor_type` so misconfiguration fails at object creation rather than silently during forward

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `PredictorHead` is available from `core.projection` and ready for import by Wave 2 agents
- 06-02-BYOL: import `from core.projection import PredictorHead` and use with `predictor_type='standard'`
- 06-03-SimSiam: import `from core.projection import PredictorHead` and use with `predictor_type='bottleneck'`
- No blockers

---
*Phase: 06-no-negative-methods*
*Completed: 2026-04-08*
