---
phase: 02-proxy-tasks-era
plan: 05
subsystem: methods
tags: [yaml-config, dispatcher, integration, smoke-test, instance-discrimination, invariant-spread]

# Dependency graph
requires:
  - phase: 02-proxy-tasks-era/02-03
    provides: InstanceDiscriminationModule with dispatcher registration
  - phase: 02-proxy-tasks-era/02-04
    provides: InvariantSpreadModule with dispatcher registration
provides:
  - Top-level methods/__init__.py that auto-registers both methods at import time
  - YAML config for Instance Discrimination (resnet18, memory bank, n_views=1)
  - YAML config for Invariant Spread (resnet18, in-batch contrastive, n_views=2)
  - End-to-end smoke tests proving YAML -> config -> dispatcher -> train pipeline
affects: [03-simclr-era, 04-moco-era]

# Tech tracking
tech-stack:
  added: []
  patterns: [yaml-config-per-method, methods-init-auto-registration, e2e-smoke-test-pattern]

key-files:
  created:
    - configs/instance_discrimination_resnet18.yaml
    - configs/invariant_spread_resnet18.yaml
  modified:
    - methods/__init__.py
    - tests/test_instance_discrimination.py
    - tests/test_invariant_spread.py

key-decisions:
  - "methods/__init__.py imports sub-packages to trigger register_method() -- no explicit registry calls at top level"
  - "YAML configs use SGD optimizer with lr=0.03 matching original paper defaults for ResNet18"

patterns-established:
  - "YAML config pattern: one config file per method+backbone combination in configs/"
  - "E2E smoke test pattern: load YAML, adapt for test, dispatcher, train 1 epoch"
  - "Registration guard pattern: check available_methods() before register_method() in tests"

requirements-completed: [ERA1-01, ERA1-02, INFRA-02]

# Metrics
duration: 4min
completed: 2026-04-02
---

# Phase 02 Plan 05: Integration Wiring and YAML Configs Summary

**Top-level methods/__init__.py auto-registration, per-method YAML configs, and end-to-end smoke tests for both Instance Discrimination and Invariant Spread**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-01T23:57:16Z
- **Completed:** 2026-04-02T00:01:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- methods/__init__.py auto-registers both methods via sub-package imports at import time
- YAML configs with documented hyperparameters (batch-size sensitivity for Invariant Spread, memory bank notes for Instance Discrimination)
- Both configs validate via TrainConfig.model_validate without error
- End-to-end smoke tests pass: YAML -> config validation -> dispatcher -> module -> trainer.fit(1 epoch)
- Full test suite green (98 tests passing)

## Task Commits

Each task was committed atomically:

1. **Task 1: Update methods/__init__.py and create YAML configs** - `bf3ef0e` (feat)
2. **Task 2: End-to-end smoke tests for both YAML configs** - `3a05ca9` (feat)

## Files Created/Modified
- `methods/__init__.py` - Auto-imports both method sub-packages to trigger dispatcher registration
- `configs/instance_discrimination_resnet18.yaml` - Instance Discrimination config (n_views=1, n_negatives=4096, SGD)
- `configs/invariant_spread_resnet18.yaml` - Invariant Spread config (n_views=2, batch_size=256, SGD)
- `tests/test_instance_discrimination.py` - Added test_yaml_config_loads_and_trains e2e smoke test
- `tests/test_invariant_spread.py` - Added test_yaml_config_loads_and_trains e2e smoke test + fixed registration guard

## Decisions Made
- methods/__init__.py imports sub-packages to trigger register_method() -- clean, no explicit registry calls at top level
- YAML configs use SGD optimizer with lr=0.03 matching original paper defaults for ResNet18
- Test registration uses available_methods() guard to avoid duplicate registration errors when clean_registry fixture preserves original state

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed dispatcher registration guard in test_dispatcher_registration**
- **Found during:** Task 2 (end-to-end smoke tests)
- **Issue:** Existing test_dispatcher_registration in test_invariant_spread.py unconditionally called register_method(), which now fails because methods/__init__.py already registers both methods at import time
- **Fix:** Added available_methods() check before register_method() call
- **Files modified:** tests/test_invariant_spread.py
- **Verification:** Full test suite passes (98 tests)
- **Committed in:** 3a05ca9 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Necessary fix for existing test compatibility with new auto-registration. No scope creep.

## Issues Encountered
None beyond the auto-fixed registration guard issue.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully wired.

## Next Phase Readiness
- Phase 02 integration complete: both methods work end-to-end from YAML config to trained model
- Both methods selectable via `method: instance_discrimination` and `method: invariant_spread` in YAML
- Ready for Phase 03 (SimCLR era) which builds on this config + dispatcher pattern

---
*Phase: 02-proxy-tasks-era*
*Completed: 2026-04-02*
