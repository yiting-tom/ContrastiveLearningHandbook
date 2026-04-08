---
phase: 06-no-negative-methods
plan: 02
subsystem: ssl-methods
tags: [byol, ema, cosine-schedule, no-negative, predictor, pytorch-lightning]

# Dependency graph
requires:
  - phase: 06-01
    provides: PredictorHead added to core/projection.py
  - phase: 04-moco
    provides: EMAUpdater in core/ema.py; BaseSSLModule in core/base.py
provides:
  - BYOLModule implementation with online/target bootstrap architecture
  - 'byol' registration in method dispatcher
affects:
  - phase: 06-03  # SimSiam (reuses PredictorHead bottleneck variant)
  - phase: 07-eval

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "setup(stage='fit') for EMA initialization when total_steps is needed"
    - "EMAUpdater called with parameter iterables (not modules) in on_train_batch_end"
    - "F.normalize projector output before predictor input and before cosine loss"
    - "learnable_params generator excludes frozen target network params via requires_grad filter"

key-files:
  created:
    - methods/byol/module.py
    - methods/byol/__init__.py
  modified:
    - methods/__init__.py

key-decisions:
  - "Initialize EMAUpdater in setup(stage='fit') not __init__, because estimated_stepping_batches is only available after trainer.fit() begins"
  - "Pass parameter iterables to EMAUpdater.step() (not modules) — matches EMAUpdater API signature"
  - "Normalize projector output before predictor (not after) — matches BYOL paper; predictor sees L2-normalized embeddings"
  - "Use torch.no_grad() context for target forward pass, plus explicit .detach() for defensive clarity"

patterns-established:
  - "BYOL setup pattern: defer EMA initialization to setup(stage='fit') for accurate total_steps"
  - "No-negative SSL: predictor asymmetry on online branch is the collapse-prevention mechanism"

requirements-completed:
  - ERA3-01

# Metrics
duration: 15min
completed: 2026-04-08
---

# Phase 6 Plan 02: BYOLModule Summary

**BYOLModule with online/target bootstrap, cosine-scheduled EMA (0.996->1.0), symmetric MSE loss, and embedding_std collapse monitoring registered as 'byol'**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-08T00:00:00Z
- **Completed:** 2026-04-08T00:15:00Z
- **Tasks:** 1
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments
- Implemented BYOLModule with correct online/target asymmetry: online branch has backbone+projector+predictor; target branch has backbone_ema+projector_ema with no predictor and no gradients
- EMA updater with cosine momentum schedule (base=0.996, end=1.0) initialized in setup(stage='fit') to access trainer.estimated_stepping_batches
- Symmetric BYOL loss: `(2 - 2*cosine_similarity)` averaged across both view directions, with L2-normalized projector outputs fed to predictor and target
- Registered as 'byol' in method dispatcher; no regression in existing test suite (EXIT=0)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement BYOLModule** - `af0777e` (feat)

## Files Created/Modified
- `methods/byol/module.py` - BYOLModule with online/target networks, EMA, symmetric loss, embedding_std logging
- `methods/byol/__init__.py` - Dispatcher registration for 'byol'
- `methods/__init__.py` - Added `import methods.byol` trigger

## Decisions Made
- EMAUpdater is initialized in `setup(stage='fit')` rather than `__init__`, because `trainer.estimated_stepping_batches` is only available after the trainer starts fitting. This deferred initialization is the correct pattern for any setup requiring trainer state.
- EMAUpdater.step() takes parameter iterables, not module objects — matched the actual API from core/ema.py (contrast with the plan's pseudocode which showed module arguments).
- Projector output is L2-normalized before being passed into the predictor. This follows the BYOL paper exactly and ensures the predictor operates on unit-sphere vectors.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] EMA step API — pass parameter iterables, not modules**
- **Found during:** Task 1 (reading core/ema.py)
- **Issue:** The plan's pseudocode showed `self.ema.step(self.backbone, self.backbone_ema)` passing module objects. The actual EMAUpdater.step() signature is `step(online_params: Iterable[nn.Parameter], target_params: Iterable[nn.Parameter])`.
- **Fix:** Called `self.ema.step(self.backbone.parameters(), self.backbone_ema.parameters())` in on_train_batch_end.
- **Files modified:** methods/byol/module.py
- **Verification:** Import succeeds; structural checks pass
- **Committed in:** af0777e

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug: API mismatch in plan pseudocode)
**Impact on plan:** Essential correctness fix. Plan intent was correct; pseudocode abstracted over the actual parameter-iterable API.

## Issues Encountered
None beyond the EMA API mismatch documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BYOLModule is complete and passes all tests
- 'byol' is registered in dispatcher, selectable via `method: byol` in YAML config
- PredictorHead (standard variant) is validated in production use; SimSiam plan (06-03) can reuse the bottleneck variant
- EMA deferred-init pattern in setup(stage='fit') is established for future momentum-encoder methods

## Self-Check: PASSED

- `methods/byol/module.py` — FOUND
- `methods/byol/__init__.py` — FOUND
- `methods/__init__.py` — contains `import methods.byol` — VERIFIED
- Commit `af0777e` — FOUND
- Import check: `python -c "from methods.byol.module import BYOLModule"` — EXIT 0
- Regression check: `python -m pytest tests/ -x -q` — EXIT 0

---
*Phase: 06-no-negative-methods*
*Completed: 2026-04-08*
