---
phase: 06-no-negative-methods
plan: "04"
subsystem: ssl-methods
tags: [simsiam, stop-gradient, no-negative, shared-encoder, bottleneck-predictor, cosine-similarity]

requires:
  - phase: 06-01
    provides: PredictorHead with predictor_type='bottleneck' (2048->512->2048)

provides:
  - SimSiamModule: shared-encoder Siamese network with stop-gradient loss
  - methods/simsiam/module.py: SimSiam CVPR 2021 implementation
  - methods/simsiam/__init__.py: register_method('simsiam') dispatcher registration

affects:
  - 06-05
  - 06-06
  - phase-07

tech-stack:
  added: []
  patterns:
    - "Shared encoder pattern: both views use same backbone+projector weights (no EMA/momentum)"
    - "Stop-gradient on z (projection side), gradient flows through p (predictor side)"
    - "Collapse monitoring via embedding_std logged every step"
    - "COLLAPSE WARNING comment co-located with .detach() calls"

key-files:
  created:
    - methods/simsiam/module.py
    - methods/simsiam/__init__.py
  modified:
    - methods/__init__.py

key-decisions:
  - "detach() applied to z (projection) not p (predictor) — matches Chen & He 2021 exactly"
  - "Projection output dim fixed at 2048 (not 128 like SimCLR) for training stability"
  - "PredictorHead(predictor_type='bottleneck', input_dim=2048, bottleneck_dim=512) from plan 06-01"
  - "No EMA or momentum encoder — stop-gradient is the only asymmetry preventing collapse"

patterns-established:
  - "Collapse warning comment co-located with .detach() call: documents failure mode at the exact line"
  - "Embedding std logged under train/embedding_std for collapse detection without additional overhead"

requirements-completed:
  - ERA3-02

duration: 8min
completed: 2026-04-08
---

# Phase 06 Plan 04: SimSiam Implementation Summary

**SimSiam shared-encoder Siamese network with symmetric stop-gradient loss (Chen & He, CVPR 2021) — no EMA, no queue, bottleneck predictor (2048->512->2048) preventing collapse via .detach() on projections**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-08T00:00:00Z
- **Completed:** 2026-04-08T00:08:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- Implemented `SimSiamModule(BaseSSLModule)` with shared encoder (single backbone + projector serving both views)
- Symmetric stop-gradient loss `-(cosim(p1, z2.detach()) + cosim(p2, z1.detach())) / 2` with COLLAPSE WARNING comment co-located at `.detach()` calls
- Collapse monitoring via `train/embedding_std = z1.std(dim=0).mean()` logged on every step under `torch.no_grad()`
- Registered as `'simsiam'` in dispatcher, importable from `methods/__init__.py`

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SimSiamModule** - `65d60ef` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `methods/simsiam/module.py` - SimSiamModule: shared encoder, 3-layer projector, bottleneck predictor, stop-gradient loss, collapse monitoring
- `methods/simsiam/__init__.py` - register_method('simsiam', SimSiamModule)
- `methods/__init__.py` - Added `import methods.simsiam` to trigger registration on package import

## Decisions Made

- `.detach()` is on `z` (projection output), not `p` (predictor output) — this matches the paper exactly. Documented with COLLAPSE WARNING comment.
- Projection output dimension is 2048 (not 128 as in SimCLR) — SimSiam paper uses higher-dim projections for stability.
- `PredictorHead(predictor_type='bottleneck', input_dim=2048, bottleneck_dim=simsiam_cfg.predictor_hidden_dim)` reuses the component from plan 06-01.
- No `backbone_ema`, no momentum encoder, no queue — stop-gradient is the only asymmetry. This is the defining characteristic of SimSiam vs BYOL.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SimSiam is registered and selectable via `method: simsiam` in YAML configs
- Stop-gradient mechanism documented with collapse warning for tutorial users
- Ready for plan 06-05 (DINO or next no-negative method in phase)

---
*Phase: 06-no-negative-methods*
*Completed: 2026-04-08*

## Self-Check: PASSED

- methods/simsiam/module.py: FOUND
- methods/simsiam/__init__.py: FOUND
- 06-04-SUMMARY.md: FOUND
- Commit 65d60ef: FOUND
