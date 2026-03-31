---
phase: 01-foundation
plan: 03
subsystem: infra
tags: [pytorch, loss, optimizer, infonce, lars, contrastive-learning, ssl]

# Dependency graph
requires: []
provides:
  - InfoNCELoss module in core/losses.py (symmetric and asymmetric modes)
  - LARS optimizer in core/optimizers.py (from scratch, tutorial-readable)
affects: [02-era1-proxy-tasks, 03-era2-moco-simclr, 04-era3-no-negatives, 05-era4-transformers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "InfoNCE symmetric mode: concatenate 2B embeddings, mask diagonal, cross-entropy"
    - "InfoNCE asymmetric mode: dot-product positive + queue negatives, cross-entropy"
    - "F.cross_entropy for numerically stable log-sum-exp (Pitfall 7)"
    - "LARS trust ratio: eta * ||p|| / ||g||, skip 1-D params for bias/norm exclusion"
    - "LARS momentum buffer maintained per-parameter in self.state"
    - "TDD red-green cycle: failing test commit then implementation commit"

key-files:
  created:
    - core/losses.py
    - core/optimizers.py
    - tests/test_losses.py
    - tests/test_optimizers.py
  modified: []

key-decisions:
  - "InfoNCELoss always L2-normalizes inputs internally — callers do not need to pre-normalize"
  - "Symmetric mode masks diagonal with -inf (not eye-based subtraction) — cleaner with F.cross_entropy"
  - "LARS implemented from scratch per D-04 — no lightly/torchlars dependency"
  - "Momentum buffer initialized from grad on first step (torch.clone) — avoids extra storage"

patterns-established:
  - "Pattern: F.cross_entropy handles log-sum-exp — never compute softmax + log manually"
  - "Pattern: LARS trust ratio skips p.ndim==1 for bias/norm exclusion (Pitfall 8)"
  - "Pattern: @torch.no_grad() on optimizer.step() — all custom optimizers follow this"

requirements-completed: [INFRA-01, INFRA-06]

# Metrics
duration: 3min
completed: 2026-03-31
---

# Phase 01 Plan 03: InfoNCELoss and LARS Optimizer Summary

**InfoNCELoss covering SimCLR symmetric and MoCo asymmetric queue modes, plus LARS optimizer with per-layer trust ratio and bias/norm exclusion — both implemented from scratch with 14 passing tests**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-31T15:42:46Z
- **Completed:** 2026-03-31T15:46:04Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `InfoNCELoss(nn.Module)` with symmetric (SimCLR/NT-Xent) and asymmetric (MoCo queue) modes, L2-normalization applied internally, `F.cross_entropy` for numerical stability
- `LARS(Optimizer)` from scratch with LARS trust ratio scaling, `exclude_bias_and_norm` for 1-D parameter exclusion, weight decay, and momentum buffer — ~60 lines, tutorial-readable
- 14 tests passing covering both modules: symmetry property, finite/positive outputs, queue mode, gradient flow, bias/norm exclusion, momentum buffer accumulation, signature check

## Task Commits

Each task was committed atomically using TDD red-green cycle:

1. **Task 1 RED: failing tests for InfoNCELoss** - `70fa1f4` (test)
2. **Task 1 GREEN: implement InfoNCELoss** - `abf064b` (feat)
3. **Task 2 RED: failing tests for LARS optimizer** - `a46da62` (test)
4. **Task 2 GREEN: implement LARS optimizer** - `06cd467` (feat)

_Note: TDD tasks have separate test and implementation commits_

## Files Created/Modified

- `core/losses.py` - InfoNCELoss with symmetric and asymmetric modes
- `core/optimizers.py` - LARS optimizer from scratch
- `tests/test_losses.py` - 7 tests for InfoNCELoss
- `tests/test_optimizers.py` - 7 tests for LARS optimizer

## Decisions Made

- `InfoNCELoss` always normalizes inputs internally — callers never need to pre-normalize, avoiding a common bug
- Symmetric mode uses `masked_fill` with diagonal mask rather than eye subtraction, which composes cleanly with `F.cross_entropy`
- LARS momentum buffer initialized via `torch.clone(grad)` on first step, matching the torchlars reference implementation
- No `refactor` phase needed — implementations were clean on first pass

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `InfoNCELoss` ready for use by SimCLR (symmetric), MoCo v1/v2 (asymmetric queue), and future methods
- `LARS` ready for use by SimCLR v1/v2, SwAV, BYOL, Barlow Twins
- Both modules export cleanly from `core/losses.py` and `core/optimizers.py`
- No blockers for subsequent plans

---
*Phase: 01-foundation*
*Completed: 2026-03-31*
