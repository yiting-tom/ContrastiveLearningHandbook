---
phase: 09-evaluation-suite
plan: "04"
subsystem: evaluation
tags: [pytorch-lightning, pytorch-grad-cam, eigencam, gradcam, fine-tuning, cam-visualization, adamw, lambda-lr]

requires:
  - phase: 09-evaluation-suite plan 01
    provides: KNNCallback (in-training eval infrastructure)
  - phase: 09-evaluation-suite plan 02
    provides: linear_probe.py (offline eval pattern established)
  - phase: 01-foundation
    provides: BaseSSLModule, FinetuneConfig, CAMConfig, EvalConfig schemas in core/config.py

provides:
  - FinetuneModule: LightningModule with 2-group AdamW + warmup-cosine LR, freeze_bn override
  - eval/finetune.py: standalone fine-tuning evaluation script (D-01 pattern)
  - get_target_layer: architecture-aware ResNet/ViT target layer selection for CAM
  - vit_reshape_transform: CLS-token removal + [B,N,D] -> [B,D,H,W] for ViT CAM
  - WrapperModule: backbone+head wrapper for GradCAM with classifier
  - eval/cam_vis.py: standalone CAM visualization script (EigenCAM default, GradCAM optional)
  - 23 unit tests covering FinetuneModule and CAM components

affects:
  - 09-evaluation-suite plan 07 (integration test uses FinetuneModule and cam_vis)

tech-stack:
  added: []
  patterns:
    - "FinetuneModule uses param_groups=[backbone, head] with AdamW + LambdaLR warmup-cosine at step interval"
    - "freeze_bn: override train() to keep BN in eval mode after super().train(mode)"
    - "CAM script defaults to EigenCAM (gradient-free) for SSL models; GradCAM only when --classifier provided"
    - "WrapperModule(backbone, head) makes GradCAM work with classifier (returns logits)"
    - "vit_reshape_transform: tensor[:, 1:, :].reshape(...).permute(0,3,1,2) pattern for ViT patch grids"
    - "run_cam caps images at cam_cfg.n_images (default 8) to prevent OOM"

key-files:
  created:
    - eval/finetune.py
    - eval/cam_vis.py
    - tests/test_eval_finetune.py
    - tests/test_eval_cam.py
  modified: []

key-decisions:
  - "Test checks LambdaLR base_lrs (not param_groups lr) because LambdaLR(lr_lambda) zeros lr at step 0 (lr_lambda(0)=0)"
  - "GradCAM without classifier falls back to EigenCAM with UserWarning rather than raising an error"
  - "load_classifier strips key prefix to extract 'weight' and 'bias' for nn.Linear state_dict compatibility"

patterns-established:
  - "FinetuneModule param_groups pattern: always list backbone and head separately for differentiated LRs"
  - "Architecture detection via 'resnet' in name.lower() and 'vit' in name.lower() - simple and sufficient"

requirements-completed: [EVAL-05, EVAL-06]

duration: ~15min
completed: 2026-04-10
---

# Phase 09 Plan 04: Fine-tuning + CAM Visualization Summary

**FinetuneModule with AdamW 2-group LR (backbone 1e-4 / head 1e-3) + freeze_bn, and CAM visualization with EigenCAM default / GradCAM+WrapperModule for SSL checkpoints**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-10T16:35:00Z
- **Completed:** 2026-04-10T16:50:17Z
- **Tasks:** 2
- **Files modified:** 4 (2 created implementation, 2 created tests)

## Accomplishments

- FinetuneModule with 2-param-group AdamW optimizer + LambdaLR warmup-cosine scheduler, configurable freeze_bn that keeps BatchNorm in eval mode during training
- CAM visualization script with architecture-aware target layer selection (ResNet layer4[-1], ViT blocks[-1].norm1), EigenCAM default for gradient-free SSL evaluation
- GradCAM path with concrete wiring: load_classifier extracts linear head weights, WrapperModule(backbone, head) returns logits for gradient flow, fallback to EigenCAM with warning when no classifier
- 23 unit tests covering all key behaviors (param groups, LRs, freeze_bn, training/validation step logging, get_target_layer, vit_reshape_transform, WrapperModule, run_cam PNG saving)

## Task Commits

Each task was committed atomically:

1. **Task 1: FinetuneModule + fine-tuning script** - `1b2054a` (feat)
2. **Task 2: CAM visualization script** - `f5d004b` (feat)

_Both tasks followed TDD: failing tests first, then implementation to pass._

## Files Created/Modified

- `eval/finetune.py` - FinetuneModule with 2-group AdamW + LambdaLR warmup-cosine, freeze_bn train() override, standalone CLI
- `eval/cam_vis.py` - get_target_layer, vit_reshape_transform, WrapperModule, run_cam, EigenCAM/GradCAM dispatch, standalone CLI
- `tests/test_eval_finetune.py` - 9 unit tests for FinetuneModule (param groups, LRs, freeze_bn, step logging)
- `tests/test_eval_cam.py` - 14 unit tests for CAM components (target layer, reshape transform, method dispatch, WrapperModule, PNG saving)

## Decisions Made

- **LambdaLR base_lrs in tests:** LambdaLR(lr_lambda) zeros the optimizer's lr at initialization (lr_lambda(0) = 0/warmup_steps = 0). Tests check `scheduler.base_lrs` instead of `optimizer.param_groups[i]["lr"]` to verify the configured LRs accurately.
- **GradCAM fallback:** When GradCAM is requested without a classifier, the script falls back to EigenCAM with a `UserWarning` rather than raising an error. This preserves usability.
- **load_classifier key stripping:** Strips prefix from state_dict keys to get `weight` and `bias` for nn.Linear compatibility, supporting both `linear.weight` and `head.weight` formats.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test adjusted to check scheduler.base_lrs instead of optimizer.param_groups lr**
- **Found during:** Task 1 (FinetuneModule test_param_group_lrs)
- **Issue:** LambdaLR initializes with lr_lambda(0) = 0 which zeros optimizer param_groups lr immediately. Test comparing `pg["lr"]` to 1e-4 fails because it's 0.0 at initialization.
- **Fix:** Tests check `scheduler.base_lrs[i]` (LambdaLR stores original lrs in `base_lrs`) to verify backbone_lr=1e-4 and head_lr=1e-3 are correctly configured.
- **Files modified:** tests/test_eval_finetune.py
- **Verification:** All 9 tests pass
- **Committed in:** 1b2054a (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - test correctness bug)
**Impact on plan:** Minor test adjustment. FinetuneModule implementation unchanged. No scope creep.

## Issues Encountered

None beyond the LambdaLR initialization behavior documented above.

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundary changes. Both scripts are offline CLI tools that read local checkpoints and files. T-09-11 (OOM from large image batches) mitigated by `cam_cfg.n_images` cap in `run_cam`.

## Known Stubs

None - both scripts are fully implemented with real logic.

## Self-Check: PASSED

- FOUND: eval/finetune.py
- FOUND: eval/cam_vis.py
- FOUND: tests/test_eval_finetune.py
- FOUND: tests/test_eval_cam.py
- FOUND: 09-04-SUMMARY.md
- FOUND: commit 1b2054a (Task 1)
- FOUND: commit f5d004b (Task 2)
- TESTS: 23/23 passing

## Next Phase Readiness

- `FinetuneModule` ready for use by plan 09-07 integration test
- `cam_vis.py` ready for use by plan 09-07 integration test
- Both scripts follow D-01 invocation pattern (config + --ckpt)

---
*Phase: 09-evaluation-suite*
*Completed: 2026-04-10*
