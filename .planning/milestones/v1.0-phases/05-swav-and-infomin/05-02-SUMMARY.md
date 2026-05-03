---
phase: 05-swav-and-infomin
plan: "02"
subsystem: methods/swav
tags: [swav, sinkhorn-knopp, optimal-transport, contrastive-learning, tdd]
dependency_graph:
  requires: []
  provides: [methods/swav/losses.py, tests/test_swav.py]
  affects: [methods/swav/module.py]
tech_stack:
  added: []
  patterns: [standalone-loss-function, torch-no-grad-decorator, injected-sinkhorn-fn]
key_files:
  created:
    - methods/swav/losses.py
    - methods/swav/__init__.py
    - tests/test_swav.py
  modified: []
decisions:
  - "Non-in-place Sinkhorn division: use Q = Q / denom.clamp(min=1e-10) instead of Q /= denom to avoid underflow NaN at n_iters > 50"
  - "Test convergence uses n_iters=100 for atol=0.05 column sum check; production default is n_iters=3 for speed"
  - "sinkhorn_fn injected as callable arg to swav_loss for testability (DummyPrototype pattern)"
metrics:
  duration_seconds: 321
  completed_date: "2026-04-08"
  tasks_completed: 1
  files_created: 3
  files_modified: 0
---

# Phase 05 Plan 02: Sinkhorn-Knopp and SwAV Loss Summary

**One-liner:** Numerically-stable Sinkhorn-Knopp optimal transport and SwAV swapped-prediction loss with injected sinkhorn callable for testability.

## What Was Built

`methods/swav/losses.py` containing:

1. `sinkhorn_knopp(scores, n_iters=3, epsilon=0.05)` — Converts raw prototype similarity scores [B, K] into a doubly-stochastic assignment matrix Q [B, K]. Decorated with `@torch.no_grad()`. Uses non-in-place division with `clamp(min=1e-10)` guards for numerical stability.

2. `swav_loss(z_list, prototype_layer, temperature, n_large_crops, sinkhorn_fn)` — Swapped-prediction loss over multi-crop views. Codes are computed from large crops only and detached (`.detach()`). All crops predict each large crop's codes.

3. `methods/swav/__init__.py` — Empty placeholder (registration added in Plan 05).

10 unit tests in `tests/test_swav.py` covering: output shape, row sum uniformity, column sum uniformity, non-negativity, multi-shape compatibility, no-grad property, doubly-stochastic property, scalar output, gradient flow through prediction side, and cross-entropy term count.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Add failing tests | 31fc598 | tests/test_swav.py |
| 1 (GREEN) | Implement sinkhorn_knopp and swav_loss | d5c8c48 | methods/swav/losses.py, methods/swav/__init__.py, tests/test_swav.py |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed plan test spec: n_iters=10 insufficient for atol=0.05 column sum convergence**
- **Found during:** Task 1 GREEN phase
- **Issue:** Plan specified `test_sinkhorn_column_sums_uniform` with `n_iters=10, atol=0.05`. With random B=64, K=100 inputs, column sum deviation reaches ~0.39 with 10 iters; 100 iters needed for atol=0.05.
- **Fix:** Updated convergence tests to use `n_iters=100`. Production default remains `n_iters=3` for training speed. Added doc note explaining the trade-off.
- **Files modified:** tests/test_swav.py
- **Commit:** d5c8c48

**2. [Rule 1 - Bug] Replaced in-place division with non-in-place for numerical stability**
- **Found during:** Task 1 GREEN phase
- **Issue:** Original plan's `Q /= Q.sum(dim=1, keepdim=True)` causes underflow to 0 followed by 0/0 = NaN at n_iters >= 50 due to accumulated floating-point precision loss.
- **Fix:** Changed to `Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-10) / K` pattern. Same algorithm, stable at any n_iters.
- **Files modified:** methods/swav/losses.py
- **Commit:** d5c8c48

## Cross-agent Interference (Out of Scope)

A parallel plan agent modified `methods/__init__.py` to import `methods.infomin` before `InfoMinConfig` was available in `core.config`. This causes `python -m pytest tests/ -x` to fail when pytest imports all test files. Isolated test `python -m pytest tests/test_swav.py -v` passes (10/10). The breakage is from incomplete infomin work in another worktree and will resolve when that plan completes.

## Known Stubs

None. `methods/swav/__init__.py` is intentionally empty — registration happens in Plan 05 (05-05-PLAN.md) when `SwAVModule` is implemented.

## Threat Flags

None. Pure numerical computation — no network endpoints, auth paths, file I/O, or external input.

## Self-Check: PASSED

- FOUND: methods/swav/losses.py
- FOUND: methods/swav/__init__.py
- FOUND: tests/test_swav.py
- FOUND: commit 31fc598 (test RED phase)
- FOUND: commit d5c8c48 (feat GREEN phase)
