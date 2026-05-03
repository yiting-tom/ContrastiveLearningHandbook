---
phase: 01-foundation
plan: "06"
subsystem: core
tags: [base-class, lightning, ssl, optimizer, scheduler, ema, logging]
dependency_graph:
  requires: [01-01, 01-02, 01-03, 01-05]
  provides: [BaseSSLModule abstract base class, public core API re-exports]
  affects: [all SSL method implementations in phases 02-10]
tech_stack:
  added: [lightning.LightningModule subclass, SequentialLR warmup-cosine scheduler]
  patterns:
    - Abstract base class with @abstractmethod on build_projector
    - Step-based LR scheduling via SequentialLR(LinearLR, CosineAnnealingLR)
    - EMA update hook in on_train_batch_end
    - try/except ImportError guards in __init__.py for parallel plan execution safety
key_files:
  created:
    - core/base.py
    - tests/test_base.py
  modified:
    - core/__init__.py
decisions:
  - "Use step-based interval for scheduler so LR updates smoothly regardless of dataset size"
  - "EMA updater wired via on_train_batch_end hook not training_step to avoid optimizer interference"
  - "try/except ImportError guards in core/__init__.py allow parallel plan execution safety"
  - "learnable_params as property (not method) so subclasses override naturally with @property decorator"
metrics:
  duration_seconds: 160
  completed_date: "2026-03-31"
  tasks_completed: 1
  files_created: 2
  files_modified: 1
---

# Phase 01 Plan 06: BaseSSLModule Abstract Base Class Summary

**One-liner:** BaseSSLModule with AdamW/SGD/LARS dispatch, step-based warmup-cosine scheduler, EMA hook via on_train_batch_end, and TensorBoard logging.

## What Was Built

`core/base.py` provides `BaseSSLModule(L.LightningModule)` — the abstract centerpiece of the framework that all SSL method subclasses extend. Method implementations only need to override `build_projector()` and `training_step()`.

### Key Components

**`core/base.py` — BaseSSLModule**
- `__init__(cfg: TrainConfig)`: stores config, calls `save_hyperparameters`, initializes EMA placeholders to None
- `build_projector()`: `@abstractmethod` — forces subclasses to declare their projection head
- `learnable_params` property: defaults to `self.parameters()`; subclasses override to exclude target network params
- `configure_optimizers()`: dispatches to `AdamW`/`SGD`/`LARS` based on `cfg.optimizer`; builds `SequentialLR(LinearLR, CosineAnnealingLR)` with step-based interval
- `on_train_batch_end()`: calls `self.ema_updater.step(online_params, target_params)` when `ema_updater` is set
- `log_train_metrics(loss, **kwargs)`: logs `train/loss` (on_step+on_epoch), `train/lr` (on_step), and any extra kwargs

**`core/__init__.py` — Public API**
Updated to re-export: `BaseSSLModule`, `build_backbone`, `TrainConfig`, `EvalConfig`, `load_config`, `ContrastiveAugmentation`, `MultiViewTransform`, `SSLDataModule`, `EMAUpdater`, `InfoNCELoss`, `LARS`, `ProjectionHead`. Each wrapped in `try/except ImportError` for parallel plan execution safety.

**`tests/test_base.py` — 12 integration tests**
- `test_subclass_trains`: minimal `DummySSLModule` trains 1 epoch on toy data
- `test_configure_optimizers_adamw/sgd/lars`: optimizer type assertions
- `test_configure_optimizers_unknown`: ValueError raised for unknown optimizer
- `test_scheduler_is_step_based`: interval == "step"
- `test_ema_hook_called`: mock verifies `EMAUpdater.step` called per batch
- `test_ema_hook_not_called_when_not_set`: no crash when ema_updater is None
- `test_learnable_params_default`: returns all module parameters
- `test_learnable_params_can_be_overridden`: subclass can narrow to exclude target network
- `test_log_train_metrics_logs_loss`: train/loss key present
- `test_log_train_metrics_logs_extra_keys`: extra kwargs logged under train/ prefix

## Verification

```
pytest tests/test_base.py -x -v   → 12 passed
python -c "from core.base import BaseSSLModule; print('base OK')"   → OK
python -c "from core import BaseSSLModule, build_backbone, ...; print('all imports OK')"   → OK
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `test_learnable_params_can_be_overridden` assertion**
- **Found during:** Task 1 test run
- **Issue:** The test asserted `len(narrow) < len(full)` but `DummySSLModule` only has `linear` (2 params) + `Identity` projector (0 params) = 2 total, same as `self.linear.parameters()`. The assertion `2 < 2` failed.
- **Fix:** Rewrote the test subclass as `WiderModule` adding a `target_linear = nn.Linear(4, 4)` with `requires_grad_(False)`, making `full` have 4 params and `narrow` have 2.
- **Files modified:** tests/test_base.py
- **Commit:** 4a23495

## Known Stubs

None — all functionality is fully wired.

## Self-Check: PASSED
