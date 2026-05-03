---
phase: 01-foundation
plan: 04
subsystem: data
tags: [torchvision, augmentation, datamodule, imagefolder, pytorch-lightning, ssl]

# Dependency graph
requires: []
provides:
  - ContrastiveAugmentation callable with strong (s=1.0, SimCLR) and weak (s=0.4, era-1) paths
  - MultiViewTransform wrapper producing configurable n_views augmented copies per image
  - ssl_collate_fn stacking multi-view batches to [n_views, B, C, H, W]
  - SSLDataModule Lightning DataModule wrapping ImageFolder with multi-view support
affects:
  - methods/simclr
  - methods/moco
  - methods/byol
  - methods/swav
  - methods/dino

# Tech tracking
tech-stack:
  added:
    - torchvision.transforms.v2 (downgraded to 0.25.0 for torch 2.10.0 compatibility)
    - lightning.LightningDataModule
    - torchvision.datasets.ImageFolder
  patterns:
    - MultiViewTransform: apply augmentation n_views times per sample, collate to [n_views, B, C, H, W]
    - ssl_collate_fn: custom collate function for SSL multi-view batches
    - ContrastiveAugmentation: strong/weak paths controlled by single `strong` boolean

key-files:
  created:
    - core/data.py
    - tests/test_data.py
  modified:
    - core/__init__.py

key-decisions:
  - "Use torchvision.transforms.v2 for all augmentations (v2 import path, not v1)"
  - "GaussianBlur kernel_size=23 (odd, 10% of 224 image size) to avoid runtime error"
  - "Strong path s=1.0 per SimCLR paper; weak path s=0.4 for era-1 methods"
  - "ssl_collate_fn uses dtype=torch.long for labels"

patterns-established:
  - "Pattern: multi-view SSL data loading via MultiViewTransform + ssl_collate_fn + SSLDataModule"
  - "Pattern: ContrastiveAugmentation strong/weak selection via single boolean flag"

requirements-completed: [FOUND-05, FOUND-06]

# Metrics
duration: 10min
completed: 2026-03-31
---

# Phase 1 Plan 4: Data Pipeline Summary

**SimCLR-strength ContrastiveAugmentation (strong/weak paths), MultiViewTransform, and SSLDataModule yielding [n_views, B, C, H, W] batches via custom collate**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-31T15:38:00Z
- **Completed:** 2026-03-31T15:48:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- ContrastiveAugmentation strong path uses s=1.0 ColorJitter, GaussianBlur (kernel=23), RandomGrayscale, RandomHorizontalFlip per SimCLR paper
- ContrastiveAugmentation weak path uses s=0.4 without GaussianBlur for era-1 methods (Instance Discrimination, etc.)
- MultiViewTransform produces exactly n_views augmented views per sample as a list
- SSLDataModule wraps ImageFolder with configurable n_views, batch_size, num_workers; auto-detects train/ subdirectory
- ssl_collate_fn stacks views list into [n_views, B, C, H, W] tensor with torch.long labels
- 11 tests pass covering all augmentation paths, n_views=2/8, labels dtype, and class discovery

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement ContrastiveAugmentation and MultiViewTransform** - `cfb3195` (feat)
2. **Task 2: Implement SSLDataModule wrapping ImageFolder** - `03ed91f` (feat)

## Files Created/Modified

- `core/data.py` - ContrastiveAugmentation, MultiViewTransform, ssl_collate_fn, SSLDataModule
- `tests/test_data.py` - 11 tests covering all augmentation and data module behavior
- `core/__init__.py` - Exports ContrastiveAugmentation, MultiViewTransform, SSLDataModule, ssl_collate_fn

## Decisions Made

- Used `v2.ToImage()` + `v2.ToDtype(torch.float32, scale=True)` instead of deprecated `v2.ToTensor()` (v2 API pattern)
- `pin_memory=True` in DataLoader (MPS devices show a non-fatal UserWarning, expected behavior)
- Val split uses center-crop single-view transform (appropriate for evaluation, not training)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed torchvision 0.22.1 incompatibility with torch 2.10.0**
- **Found during:** Task 1 setup
- **Issue:** torchvision 0.22.1 (latest) raised `RuntimeError: operator torchvision::nms does not exist` when imported with torch 2.10.0
- **Fix:** Downgraded torchvision to 0.25.0 which is compatible with torch 2.10.0
- **Files modified:** system packages (pip)
- **Verification:** `python -c "from torchvision.transforms import v2; print('v2 OK')"` exits 0
- **Committed in:** cfb3195 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required to unblock all torchvision imports. No scope creep.

## Issues Encountered

None beyond the torchvision version fix documented above.

## Known Stubs

None - all data pipeline components are fully wired and tested.

## Next Phase Readiness

- SSL data pipeline fully operational; any method phase can use `SSLDataModule` immediately
- `from core.data import ContrastiveAugmentation, MultiViewTransform, SSLDataModule` works
- n_views=2 (SimCLR/MoCo), n_views=8+ (SwAV/DINO) both verified by tests
- Strong augmentation path matches SimCLR paper specification

---
*Phase: 01-foundation*
*Completed: 2026-03-31*
