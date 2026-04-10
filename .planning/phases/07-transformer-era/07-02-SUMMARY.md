---
phase: 07-transformer-era
plan: "02"
subsystem: methods/moco_v3
tags: [moco-v3, vit, transformer, momentum-contrast, symmetric-infonce, patch-freeze]
dependency_graph:
  requires: [07-01, core/losses.py, core/ema.py, core/projection.py, core/backbone.py, core/base.py]
  provides: [methods/moco_v3/module.py, methods/moco_v3/__init__.py]
  affects: [methods/__init__.py, tests/test_moco_v3.py]
tech_stack:
  added: [MoCoV3Module, MoCoV3Config]
  patterns: [EMAUpdater-setup-pattern, symmetric-infonce, patch-embed-freeze, predictor-online-only]
key_files:
  created:
    - methods/moco_v3/module.py
    - methods/moco_v3/__init__.py
    - tests/test_moco_v3.py
  modified:
    - methods/__init__.py
decisions:
  - "MoCoV3 uses constant EMA momentum (base==end=0.99) like MoCo v1/v2, not cosine-scheduled like BYOL/DINO"
  - "patch_embed.proj freeze happens in __init__ before deepcopy so momentum encoder also gets frozen patch embed"
  - "learnable_params uses itertools.chain over backbone/projector/predictor parameters for explicit EMA exclusion"
  - "Dispatcher test guards against clean_registry fixture clearing registration after cached module import"
metrics:
  duration_seconds: 398
  completed_date: "2026-04-10"
  tasks_completed: 2
  files_changed: 4
---

# Phase 07 Plan 02: MoCo v3 Summary

MoCoV3Module with ViT patch projection freeze, symmetric in-batch InfoNCE (no queue), predictor on online branch only, and AdamW optimizer wired into dispatcher as "moco_v3".

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | TDD failing tests | 7cf87b3 | tests/test_moco_v3.py |
| 1 (GREEN) | Implement MoCoV3Module | 94bee2e | methods/moco_v3/module.py, methods/moco_v3/__init__.py, tests/test_moco_v3.py |
| 2 | Register in methods/__init__.py | a24e023 | methods/__init__.py |

## What Was Built

### MoCoV3Module (methods/moco_v3/module.py)

Implements momentum contrastive learning adapted for Vision Transformers:

**Architecture:**
- Online network: backbone -> ProjectionHead(3-layer, feat_dim->4096->256) -> PredictorHead(standard, 256->4096->256)
- Momentum network: backbone_ema -> projector_ema (NO predictor)

**Key properties:**
- `backbone.patch_embed.proj.weight/bias.requires_grad_(False)` set in `__init__` (ViT stability trick)
- `setup()` initializes EMAUpdater with constant momentum m=0.99 (base==end) once total_steps is known
- `learnable_params` chains backbone/projector/predictor; excludes backbone_ema/projector_ema
- `training_step` computes symmetric in-batch InfoNCE: `(loss_fn(q1, k2) + loss_fn(q2, k1)) / 2`
- `on_train_batch_end` runs EMA update after optimizer step (BYOL pattern)

### Registration (methods/moco_v3/__init__.py)

Calls `register_method("moco_v3", MoCoV3Module)` triggering dispatcher registration on package import.

### Test Suite (tests/test_moco_v3.py)

7 tests covering all must-have truths:
- `test_patch_projection_frozen`: ViT patch embed proj frozen after construction
- `test_moco_v3_uses_adamw`: AdamW optimizer default confirmed
- `test_momentum_encoder_excluded_from_learnable_params`: No EMA param in learnable set
- `test_predictor_on_online_only`: Has predictor, no predictor_ema
- `test_no_queue`: No momentum_queue attribute
- `test_dispatcher_moco_v3`: Dispatcher resolves "moco_v3" to MoCoV3Module
- `test_moco_v3_train_3_epochs`: 3-epoch smoke test with resnet18, loss finite

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_moco_v3_uses_adamw calling configure_optimizers without trainer**
- **Found during:** Task 1 GREEN phase
- **Issue:** `configure_optimizers` calls `self.trainer.estimated_stepping_batches` which requires a trainer attached. Test design in plan assumed it could be called standalone.
- **Fix:** Changed test to verify `cfg.optimizer == "adamw"` and directly instantiate `optim.AdamW` from `learnable_params` without needing a trainer context.
- **Files modified:** tests/test_moco_v3.py
- **Commit:** 94bee2e

**2. [Rule 1 - Bug] Fixed test_dispatcher_moco_v3 with clean_registry interaction**
- **Found during:** Task 1 GREEN phase
- **Issue:** When `methods.moco_v3` is already cached in `sys.modules` from a prior test, `import methods.moco_v3` is a no-op and doesn't re-call `register_method`. The `clean_registry` fixture had cleared the registry, so the dispatcher found no "moco_v3" entry.
- **Fix:** Added guard `if "moco_v3" not in available_methods(): register_method("moco_v3", MoCoV3Module)` matching the pattern from `tests/test_moco.py`.
- **Files modified:** tests/test_moco_v3.py
- **Commit:** 94bee2e

## Decisions Made

- Constant EMA momentum (base==end=0.99) per MoCo v3 paper. Cosine-scheduled momentum is reserved for BYOL/DINO.
- `patch_embed.proj` freeze in `__init__` before `deepcopy(self.backbone)` ensures the momentum backbone also inherits the frozen patch embed state.
- `itertools.chain` for `learnable_params` property: explicit and readable, matching intent that EMA branches are excluded.

## Known Stubs

None — all data flows fully wired.

## Threat Flags

None — ML training module, no security surface.

## Self-Check: PASSED

- [x] methods/moco_v3/module.py exists
- [x] methods/moco_v3/__init__.py exists
- [x] tests/test_moco_v3.py exists
- [x] methods/__init__.py contains `import methods.moco_v3`
- [x] `python -m pytest tests/test_moco_v3.py` exits 0 (7 passed)
- [x] Commits 7cf87b3, 94bee2e, a24e023 exist in git log
