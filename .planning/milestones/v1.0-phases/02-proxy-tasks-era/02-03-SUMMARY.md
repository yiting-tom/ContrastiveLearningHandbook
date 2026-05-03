---
phase: 02-proxy-tasks-era
plan: 03
subsystem: methods
tags: [instance-discrimination, memory-bank, nce-loss, ssl, contrastive-learning]

# Dependency graph
requires:
  - phase: 02-01
    provides: MemoryBank and InstanceDiscriminationConfig
  - phase: 02-02
    provides: NCELossWithFixedZ
provides:
  - InstanceDiscriminationModule(BaseSSLModule) -- complete ERA1-01 method
  - IndexedDataset wrapper for sample-index-aware dataloading
  - ssl_collate_with_index collate function for (views, labels, indices) batches
  - Dispatcher registration as 'instance_discrimination'
affects: [02-04, 02-05, phase-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy-memory-bank-init, indexed-dataset-wrapper, learnable-params-exclusion]

key-files:
  created:
    - methods/instance_discrimination/module.py
    - tests/test_instance_discrimination.py
  modified:
    - core/data.py
    - methods/instance_discrimination/__init__.py

key-decisions:
  - "Memory bank initialized lazily (set externally in tests, or via setup() when trainer.datamodule is available)"
  - "learnable_params uses itertools.chain over backbone + projector params, excluding memory bank and NCE loss"
  - "Negative sampling uses simple global random sampling (no exclusion of current batch indices)"

patterns-established:
  - "IndexedDataset wrapper pattern: wrap base dataset to append sample index to each tuple"
  - "ssl_collate_with_index: collate function returning (views, labels, indices) 3-tuple"
  - "learnable_params override: chain backbone + projector params to exclude non-learnable modules"

requirements-completed: [ERA1-01]

# Metrics
duration: 5min
completed: 2026-04-01
---

# Phase 02 Plan 03: InstanceDiscriminationModule Summary

**Non-parametric Instance Discrimination module wired to MemoryBank + NCELossWithFixedZ, with IndexedDataset for sample-index-aware dataloading**

## Performance

- **Duration:** 5 min (326 seconds)
- **Started:** 2026-04-01T16:18:18Z
- **Completed:** 2026-04-01T16:23:44Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- IndexedDataset and ssl_collate_with_index added to core/data.py for memory-bank methods
- InstanceDiscriminationModule implements full pipeline: backbone -> projector -> L2-normalize -> NCE loss with bank
- Memory bank updated each step with current encoder outputs; Z fixed after first mini-batch
- Registered as 'instance_discrimination' in method dispatcher
- 5 integration tests passing: train_5_epochs, z_fixed, dispatcher, learnable_params, bank_updated

## Task Commits

Each task was committed atomically:

1. **Task 1: Add IndexedDataset and ssl_collate_with_index** - `fa48534` (feat)
2. **Task 2 RED: Failing tests for InstanceDiscriminationModule** - `d554bf7` (test)
3. **Task 2 GREEN: Implement InstanceDiscriminationModule + dispatcher** - `87cc655` (feat)

## Files Created/Modified
- `core/data.py` - Added IndexedDataset wrapper and ssl_collate_with_index collate function
- `methods/instance_discrimination/module.py` - InstanceDiscriminationModule(BaseSSLModule) with full training pipeline
- `methods/instance_discrimination/__init__.py` - Dispatcher registration as 'instance_discrimination'
- `tests/test_instance_discrimination.py` - 5 integration/unit tests for the module

## Decisions Made
- Memory bank initialized lazily -- set externally in tests, or via setup() with trainer.datamodule
- learnable_params uses itertools.chain over backbone + projector, excluding bank and NCE loss buffers
- Simple global random negative sampling (no batch-index exclusion) -- matches paper's approach

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed dispatcher registration test for clean_registry compatibility**
- **Found during:** Task 2 GREEN (test execution)
- **Issue:** clean_registry fixture saves/restores registry state, but import-time register_method only fires once per process. When full test suite runs, the registration already exists in saved state, causing duplicate registration error.
- **Fix:** Changed test to check available_methods() before calling register_method, avoiding duplicate registration.
- **Files modified:** tests/test_instance_discrimination.py
- **Verification:** Full test suite passes (91 tests)
- **Committed in:** 87cc655 (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Minor test fixture compatibility fix. No scope creep.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- InstanceDiscriminationModule complete and tested, ready for integration
- IndexedDataset and ssl_collate_with_index available for future memory-bank methods (CMC, etc.)
- Plans 02-04 (InvariantSpread) and 02-05 can proceed

---
*Phase: 02-proxy-tasks-era*
*Completed: 2026-04-01*
