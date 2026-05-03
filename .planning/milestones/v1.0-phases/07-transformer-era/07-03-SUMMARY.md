---
phase: 07-transformer-era
plan: "03"
subsystem: methods/dino
tags: [dino, self-distillation, centering, multi-crop, ema, vit]
dependency_graph:
  requires: [07-01, 07-02]
  provides: [methods/dino/module.py, methods/dino/__init__.py, tests/test_dino.py]
  affects: [methods/__init__.py, core/dispatcher.py]
tech_stack:
  added: []
  patterns:
    - DINOModule student-teacher self-distillation with centering + sharpening
    - Centering buffer updated BEFORE loss computation (D-02 critical ordering)
    - Teacher sees global crops only; student sees all crops (multi-crop)
    - EMA cosine-scheduled momentum 0.996->1.0 (shared pattern with BYOL/MoCo v3)
    - Teacher temperature linear warmup: 0.07->0.04 over warmup_teacher_temp_epochs
    - prototype_layer: nn.Linear(256, 65536, bias=False) outside ProjectionHead
key_files:
  created:
    - methods/dino/module.py
    - methods/dino/__init__.py
    - tests/test_dino.py
  modified:
    - methods/__init__.py
decisions:
  - "DINOModule uses itertools.chain for learnable_params to explicitly exclude all _ema params"
  - "Centering update uses self.center = assignment (not in-place) to avoid autograd issues"
  - "test_dispatcher_dino follows moco_v3 guard pattern: re-register if clean_registry cleared it"
  - "Smoke test uses n_prototypes=128 for speed instead of 65536"
metrics:
  duration_seconds: 305
  completed_date: "2026-04-10"
  tasks_completed: 2
  files_created: 3
  files_modified: 1
---

# Phase 7 Plan 3: DINO Module Summary

**One-liner:** DINOModule implementing student-teacher self-distillation with centering+sharpening, multi-crop support (teacher global-only), cosine EMA 0.996->1.0, and 65536-prototype cross-entropy loss.

## What Was Built

DINOModule implementing the DINO (Caron et al., ICCV 2021) self-supervised learning method:

- **Student network**: backbone -> 3-layer ProjectionHead (feat_dim->2048->256) -> nn.Linear(256, 65536, bias=False)
- **Teacher (EMA) network**: deep copies of all three student components, frozen (requires_grad=False)
- **Centering buffer**: `register_buffer('center', torch.zeros(n_prototypes))` updated BEFORE loss each step
- **Multi-crop**: teacher processes only first `n_global=2` crops; student processes all crops
- **Loss**: cross-entropy between centered+sharpened teacher softmax and student log-softmax, skipping same-view pairs
- **EMA**: cosine-scheduled momentum 0.996->1.0 via EMAUpdater, applied in `on_train_batch_end`
- **Teacher temp warmup**: linear warmup 0.07->0.04 over `warmup_teacher_temp_epochs`
- **Dispatcher**: registered as "dino" in both `methods/dino/__init__.py` and `methods/__init__.py`

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| TDD RED | Failing tests for DINOModule | 676aa2c | tests/test_dino.py |
| Task 1 GREEN | DINOModule implementation | 003f497 | methods/dino/__init__.py, methods/dino/module.py, tests/test_dino.py |
| Task 2 | Register dino in methods/__init__.py | 5af2ac9 | methods/__init__.py |

## Tests

All 8 tests pass (`python -m pytest tests/test_dino.py -x -q`):

- `test_centering_buffer_exists`: center buffer shape [n_prototypes], registered buffer
- `test_centering_update_before_loss`: center non-zero after training_step (zero at init)
- `test_teacher_global_crops_only`: backbone_ema called exactly 2 times for 4-crop batch
- `test_teacher_no_predictor`: no `predictor_ema` or `teacher_predictor` attribute
- `test_momentum_encoder_excluded`: zero overlap between learnable_params and EMA params
- `test_prototype_output_dim`: prototype_layer out_features=65536, bias=None
- `test_dispatcher_dino`: dispatcher returns DINOModule instance
- `test_dino_train_3_epochs`: 3-epoch smoke test, loss finite and not diverging

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_dispatcher_dino used wrong import pattern**
- **Found during:** Task 1 GREEN testing
- **Issue:** Test used `import methods` which is a no-op after first import, so `clean_registry` fixture restored an empty registry and the dispatcher couldn't find "dino"
- **Fix:** Followed moco_v3 pattern: `import methods.dino` + guard `if "dino" not in available_methods(): register_method("dino", DINOModule)`
- **Files modified:** tests/test_dino.py
- **Commit:** 003f497

## Known Stubs

None — all data flows are wired through real MultiCropDataset and SSLDataModule.

## Threat Flags

None — ML training module with no security surface.

## Self-Check: PASSED

- [x] methods/dino/module.py exists and contains `class DINOModule(BaseSSLModule):`
- [x] methods/dino/__init__.py contains `register_method("dino", DINOModule)`
- [x] tests/test_dino.py contains all 8 required tests
- [x] methods/__init__.py contains `import methods.dino`
- [x] Commits 676aa2c, 003f497, 5af2ac9 exist in git log
- [x] All 8 tests pass
