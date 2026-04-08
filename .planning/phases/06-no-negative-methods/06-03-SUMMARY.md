---
phase: 06-no-negative-methods
plan: "03"
subsystem: testing
tags: [byol, stop-gradient, unit-test, pytorch, lightning, ema]

requires:
  - phase: 06-02
    provides: BYOLModule with backbone_ema, projector_ema, predictor, and training_step

provides:
  - Stop-gradient validation tests that confirm target branch receives zero gradient
  - Stop-gradient comment in source code documenting the collapse-prevention mechanism
  - Five unit tests covering instantiation, gradient flow, loss finiteness, and frozen params

affects:
  - 06-04
  - 06-05
  - Any phase building on BYOL or teaching stop-gradient mechanics

tech-stack:
  added: []
  patterns:
    - "Mock self.log and self.log_train_metrics for trainer-free Lightning unit tests"
    - "Call backward() on training_step output to verify gradient isolation"

key-files:
  created:
    - tests/test_byol.py
  modified:
    - methods/byol/module.py

key-decisions:
  - "Mock self.log / self.log_train_metrics rather than spinning up a full Lightning Trainer for gradient-isolation tests — faster and avoids trainer state side-effects"
  - "Use torch.no_grad() context (already present) plus .detach() as dual stop-gradient guards; comment documents why both are present"

patterns-established:
  - "run_training_step helper pattern: patches instance-level log methods, enabling direct training_step calls without a trainer"
  - "Gradient-isolation test pattern: call loss.backward() then check target param .grad is None or zero"

requirements-completed:
  - ERA3-01

duration: 12min
completed: 2026-04-08
---

# Phase 06 Plan 03: BYOL Stop-Gradient Validation Summary

**Five-test stop-gradient validation suite proving target branch receives zero gradient; source comment documents why removing .detach() causes collapse.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-08T00:00:00Z
- **Completed:** 2026-04-08T00:12:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Added a 5-line stop-gradient comment in `methods/byol/module.py` at the `.detach()` call site in `training_step`, explaining the collapse-prevention mechanism and warning that removing it causes immediate collapse
- Created `tests/test_byol.py` with 5 tests: `test_byol_target_zero_grad` (core correctness), `test_byol_online_params_have_grad`, `test_byol_instantiation`, `test_byol_training_step_returns_finite_loss`, `test_byol_target_frozen_at_init`
- All 5 new tests pass; no regressions (207 total tests pass)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add stop-gradient comment and write validation tests** - `74d04e6` (feat)

## Files Created/Modified

- `tests/test_byol.py` - Five unit tests for BYOL stop-gradient correctness, gradient flow, instantiation, loss finiteness, and target parameter freezing
- `methods/byol/module.py` - Added 5-line stop-gradient comment at `.detach()` call site in `training_step`

## Decisions Made

- Used `unittest.mock.MagicMock` to patch `self.log` and `self.log_train_metrics` at the instance level, enabling `training_step` to be called directly without a Lightning Trainer. This avoids `_call_lightning_module_hook` errors and keeps the test focused on gradient flow, not trainer bookkeeping.
- Kept both the `torch.no_grad()` context and `t1.detach()` / `t2.detach()` in the loss, and added the comment explaining why both guards coexist: `no_grad` blocks gradient accumulation at the tensor level; `detach()` makes intent explicit in the loss expression and guards against accidental gradient leakage if the no_grad context is ever refactored.

## Deviations from Plan

None - plan executed exactly as written. The `self.ema is None` guard was already in place in `on_train_batch_end` and the `training_step` does not call EMA directly, so no module changes beyond the comment were required.

## Issues Encountered

None. The plan correctly anticipated that `self.log()` would need mocking for trainer-free tests. Implementing the `run_training_step` helper with instance-level mocks resolved this cleanly on the first attempt.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Stop-gradient mechanism is verified by automated tests — ERA3-01 correctness guarantee is satisfied
- `tests/test_byol.py` is available as a regression guard for any future refactoring of the BYOL training loop
- Ready for 06-04 (next no-negative method in the phase)

---
*Phase: 06-no-negative-methods*
*Completed: 2026-04-08*

## Self-Check: PASSED

- FOUND: tests/test_byol.py
- FOUND: methods/byol/module.py
- FOUND commit: 74d04e6 feat(06-03): add stop-gradient comment and BYOL validation tests
