---
phase: 01-foundation
plan: 02
subsystem: infra
tags: [timm, pytorch, backbone, projection-head, ssl, resnet, vit, batch-norm]

# Dependency graph
requires:
  - phase: 01-foundation/01-01
    provides: "project scaffold, core/ package structure, __init__.py"
provides:
  - "build_backbone(model_name, pretrained) factory using timm — returns (backbone, feat_dim)"
  - "ProjectionHead MLP with configurable num_layers, BN+ReLU intermediate, BN-only final"
  - "test_backbone.py: 6 tests covering resnet50, vit_small, forward shape, error handling"
  - "test_projection.py: 6 tests covering BN+ReLU pattern, shapes, use_bn=False, param count"
affects:
  - 01-foundation/01-05
  - methods/simclr
  - methods/moco
  - methods/byol
  - methods/barlow-twins
  - methods/swav
  - methods/dino

# Tech tracking
tech-stack:
  added: [timm 1.0.19]
  patterns:
    - "timm.create_model(model_name, pretrained=pretrained, num_classes=0) — always num_classes=0 for backbone"
    - "feat_dim = backbone.num_features — never hardcode feature dimensions"
    - "ProjectionHead: Linear -> [BN -> ReLU] intermediate, Linear -> [BN] final (no ReLU)"
    - "TDD with RED (failing test commit) then GREEN (implementation commit)"

key-files:
  created:
    - core/backbone.py
    - core/projection.py
    - tests/test_backbone.py
    - tests/test_projection.py
  modified: []

key-decisions:
  - "build_backbone always uses backbone.num_features — never hardcode 2048/384/512 etc."
  - "ProjectionHead requires at least 2 layers (assertion) — matches SSL literature minimum"
  - "Final projection layer has BN but no ReLU — matches SimCLR/MoCo/BYOL conventions"
  - "use_bn=False option retained for DINO/methods that skip batch norm in projector"

patterns-established:
  - "Pattern: Backbone factory always returns (backbone, feat_dim) tuple — callers never inspect internals"
  - "Pattern: ProjectionHead is model-agnostic — accepts any input_dim, used by all SSL methods"
  - "Pattern: TDD RED → commit → GREEN → commit for each feature"

requirements-completed: [FOUND-03, FOUND-04]

# Metrics
duration: 15min
completed: 2026-03-31
---

# Phase 01 Plan 02: Backbone Factory and Projection Head Summary

**timm backbone factory and reusable ProjectionHead MLP — foundation for all SSL method implementations**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-31T00:00:00Z
- **Completed:** 2026-03-31
- **Tasks:** 2 (each with TDD RED + GREEN commits)
- **Files modified:** 4 new files

## Accomplishments

- `build_backbone(model_name, pretrained=False)` factory using `timm.create_model(..., num_classes=0)` — works with any timm model, returns `(backbone, feat_dim)` with `feat_dim = backbone.num_features`
- `ProjectionHead` MLP module: configurable `num_layers`, `hidden_dim`, `output_dim`, `use_bn` — BN+ReLU on intermediate layers, BN-only on final layer
- Full test coverage: 6 backbone tests (parametrized over ResNet50 and ViT-Small) + 6 projection tests (BN pattern, shapes, no-BN mode, param count)

## Task Commits

Each task was committed atomically using TDD:

1. **Task 1 RED: build_backbone failing tests** - `2f36a44` (test)
2. **Task 1 GREEN: build_backbone implementation** - `95a5161` (feat)
3. **Task 2 RED: ProjectionHead failing tests** - `7684c73` (test)
4. **Task 2 GREEN: ProjectionHead implementation** - `8567b21` (feat)

_Note: TDD tasks have separate RED (test) and GREEN (implementation) commits_

## Files Created/Modified

- `core/backbone.py` - `build_backbone(model_name, pretrained)` factory wrapping timm
- `core/projection.py` - `ProjectionHead(nn.Module)` reusable MLP with BN+ReLU pattern
- `tests/test_backbone.py` - 6 tests: feat_dim values, no classifier head, forward shapes, unknown model error
- `tests/test_projection.py` - 6 tests: BN+ReLU pattern (2-layer and 3-layer), shapes, use_bn=False, param count

## Decisions Made

- `feat_dim` is always read from `backbone.num_features`, never hardcoded — future-proof for any timm model
- `ProjectionHead` enforces `num_layers >= 2` via assertion — matches SSL literature minimum (all methods use at least 2 layers)
- Final layer has BN but no ReLU — matches SimCLR, MoCo v2, BYOL, and Barlow Twins conventions
- `use_bn=False` option included for DINO and other methods that use layer norm or no normalization in the projector

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] torchvision/torch version incompatibility**
- **Found during:** Task 1 (running RED tests)
- **Issue:** Installed torchvision 0.22.1 was incompatible with torch 2.10.0 — `operator torchvision::nms does not exist` error prevented timm import
- **Fix:** Reinstalled torchvision 0.25.0 which is compatible with torch 2.10.0
- **Files modified:** None (environment fix only)
- **Verification:** `import timm` and `pytest tests/test_backbone.py` succeeded after fix
- **Committed in:** N/A (environment only, no code change)

---

**Total deviations:** 1 auto-fixed (1 blocking environment issue)
**Impact on plan:** Environment fix was necessary to unblock test execution. No scope creep.

## Issues Encountered

- torchvision version mismatch with installed torch — resolved by reinstalling compatible torchvision (Rule 3 auto-fix)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `build_backbone` and `ProjectionHead` are ready to be used by `BaseSSLModule` (plan 01-05) and all method implementations
- Both components are independently tested and have stable APIs
- No blockers for subsequent plans

---
*Phase: 01-foundation*
*Completed: 2026-03-31*
