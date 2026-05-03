---
phase: 05-swav-and-infomin
plan: 05
subsystem: ssl-methods
tags: [swav, pytorch-lightning, sinkhorn-knopp, prototype-layer, multi-crop, contrastive-learning]

requires:
  - phase: 05-swav-and-infomin
    plan: 01
    provides: MultiCropDataset and SSLDataModule with multi-crop collate
  - phase: 05-swav-and-infomin
    plan: 02
    provides: sinkhorn_knopp and swav_loss from methods/swav/losses.py
  - phase: 05-swav-and-infomin
    plan: 03
    provides: PrototypeLayer with normalize_prototypes and zero_prototype_gradients
  - phase: 05-swav-and-infomin
    plan: 04
    provides: SwAVConfig fields in core/config.py

provides:
  - SwAVModule(BaseSSLModule) integrating MultiCropDataset, PrototypeLayer, Sinkhorn-Knopp, and swapped-prediction loss
  - Dispatcher registration for 'swav' method key
  - Training tests: 5-epoch convergence, prototype normalization, learnable_params, freeze logic
  - Numerical stability fix for sinkhorn_knopp (log-sum-exp trick prevents float32 overflow)

affects:
  - Phase 06 (DINO) — reuses MultiCropDataset and PrototypeLayer patterns established here
  - Phase 08 (evaluation) — SwAVModule is selectable via method: swav in YAML config

tech-stack:
  added: []
  patterns:
    - itertools.chain in learnable_params to include backbone + projector + prototype layer
    - on_train_batch_end calls super() then normalize_prototypes (EMA-safe order)
    - on_before_optimizer_step zeros prototype gradients during freeze epochs
    - functools.partial for sinkhorn_fn injection into swav_loss
    - TDD: RED commit (failing tests) -> GREEN commit (implementation + fix)

key-files:
  created:
    - methods/swav/module.py
    - methods/swav/__init__.py (updated from placeholder)
  modified:
    - methods/__init__.py
    - methods/swav/losses.py
    - tests/test_swav.py

key-decisions:
  - "learnable_params uses itertools.chain to return an iterator (not a list) of backbone+projector+prototype params, consistent with configure_optimizers calling list()"
  - "on_train_batch_end calls super() first then normalize_prototypes — super() handles EMA, normalization must happen after optimizer step"
  - "sinkhorn_knopp numerical stability fix: subtract per-prototype column max before exp to prevent float32 overflow (epsilon=0.05 with any score > 4.4 caused NaN)"
  - "Dispatcher registration triggered by import methods.swav in methods/__init__.py auto-import chain"

patterns-established:
  - "Multi-crop training_step: encode all crops in loop, compute swav_loss with partial sinkhorn_fn injection"
  - "Prototype freeze hook: on_before_optimizer_step checks PrototypeLayer.should_freeze_prototypes(epoch, cfg)"
  - "Test isolation: clean_registry autouse fixture per test file, not conftest, to avoid cross-file interference"

requirements-completed:
  - ERA2-05
  - INFRA-04

duration: ~20min
completed: 2026-04-08
---

# Phase 05 Plan 05: SwAV Module Summary

**SwAVModule integrating multi-crop encoding, Sinkhorn-Knopp code assignment, swapped-prediction loss, prototype normalization, and epoch-based freeze logic; registered as 'swav' in the method dispatcher**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-04-08T~T12:00Z
- **Completed:** 2026-04-08
- **Tasks:** 2 (1 standard + 1 TDD)
- **Files modified:** 5

## Accomplishments
- Implemented `SwAVModule(BaseSSLModule)` integrating all Phase 05 components (Plans 01-04)
- Registered SwAV as `'swav'` in the method dispatcher via `methods/swav/__init__.py`
- Added 6 integration tests covering dispatcher registration, 5-epoch training, prototype normalization, learnable_params, and prototype freeze
- Fixed pre-existing numerical stability bug in `sinkhorn_knopp` (float32 overflow at epsilon=0.05 with large scores)
- Full test suite passes: 191/191 tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SwAVModule with full training pipeline** - `475b7c5` (feat)
2. **Task 2 (RED): Add failing tests for dispatcher and training** - `98379d2` (test)
3. **Task 2 (GREEN): Register SwAV in dispatcher and fix sinkhorn stability** - `95985e1` (feat)

## Files Created/Modified
- `methods/swav/module.py` - SwAVModule class with training pipeline hooks
- `methods/swav/__init__.py` - Dispatcher registration (register_method("swav", SwAVModule))
- `methods/__init__.py` - Added `import methods.swav` to auto-import chain
- `methods/swav/losses.py` - Fixed sinkhorn_knopp float32 overflow (log-sum-exp stability)
- `tests/test_swav.py` - Added 6 integration tests (dispatcher, training, normalization, freeze)

## Decisions Made
- `learnable_params` uses `itertools.chain` to return an iterator (consistent with `list(self.learnable_params)` in `configure_optimizers`)
- `on_train_batch_end` calls `super()` first then `normalize_prototypes()` to maintain EMA-safe ordering
- `functools.partial` used to inject `sinkhorn_fn` with pre-bound `n_iters` and `epsilon` into `swav_loss`
- Numerical stability fix applied to `sinkhorn_knopp`: subtract per-column max before exp — this is mathematically equivalent and prevents float32 overflow when epsilon is small

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed sinkhorn_knopp float32 numerical overflow**
- **Found during:** Task 2 (GREEN phase - running full test suite)
- **Issue:** `torch.exp(scores / epsilon)` overflows float32 when epsilon=0.05 and any score > ~4.4 (since exp(4.4/0.05) = exp(88) exceeds float32 max). This caused `NaN` in `test_sinkhorn_row_sums_uniform` when run after `test_simclr.py`'s training tests advanced the RNG to a state producing out-of-range values.
- **Fix:** Added per-column max subtraction before exp: `scaled = scaled - scaled.max(dim=0, keepdim=True).values`. This is the standard log-sum-exp stability trick — mathematically equivalent since subsequent normalization is scale-invariant.
- **Files modified:** `methods/swav/losses.py`
- **Verification:** 191/191 tests pass including full suite; previously 1 failed
- **Committed in:** `95985e1` (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - numerical stability bug)
**Impact on plan:** Essential correctness fix — sinkhorn was silently failing for real training data. No scope creep.

## Issues Encountered
- `test_sinkhorn_row_sums_uniform` failed only when run after `test_simclr.py` because the RNG state advanced to values causing exp overflow. Root cause was missing numerical stability in sinkhorn_knopp (from Plan 02), triggered by this plan's test isolation analysis.

## Next Phase Readiness
- SwAV is fully operational and selectable via `method: swav` in YAML config
- Prototype layer tested and confirmed L2-normalized post-training
- sinkhorn_knopp is now numerically stable for arbitrary input magnitudes
- Phase 06 (DINO) can reuse MultiCropDataset and PrototypeLayer patterns directly

---
*Phase: 05-swav-and-infomin*
*Completed: 2026-04-08*
