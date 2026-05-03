---
phase: 05-swav-and-infomin
plan: "01"
subsystem: data-pipeline
tags: [multi-crop, swav, dino, infra-04, dataset-wrapper]
dependency_graph:
  requires: [core/data.py (ContrastiveAugmentation, SSLDataModule)]
  provides: [MultiCropDataset, ssl_collate_multi_crop, SSLDataModule.dataset param]
  affects: [Phase 07 DINO (will reuse MultiCropDataset directly)]
tech_stack:
  added: []
  patterns: [TDD, dataset-wrapper, collate-override]
key_files:
  created: [tests/test_multi_crop.py]
  modified: [core/data.py]
decisions:
  - "MultiCropDataset takes no-transform base dataset and applies ContrastiveAugmentation internally -- base dataset must return (PIL_Image, label)"
  - "ssl_collate_multi_crop returns a list of tensors (not stacked) because large/small crops have incompatible spatial dims"
  - "SSLDataModule.dataset parameter accepted as pre-wrapped dataset, bypasses ImageFolder creation"
  - "SSLDataModule.train_dataloader uses isinstance(MultiCropDataset) check to select collate_fn automatically"
metrics:
  duration_seconds: 632
  completed_date: "2026-04-08"
  tasks_completed: 1
  files_modified: 2
---

# Phase 05 Plan 01: MultiCropDataset Implementation Summary

**One-liner:** MultiCropDataset wrapper producing n_large 224x224 + n_small 96x96 crops per sample via ContrastiveAugmentation, with SSLDataModule auto-detection for SwAV/DINO reuse.

## What Was Built

Added `MultiCropDataset`, `ssl_collate_multi_crop`, and updated `SSLDataModule` in `core/data.py` to support variable-size multi-crop augmentation required by SwAV (Phase 5) and DINO (Phase 7).

### MultiCropDataset

`MultiCropDataset(dataset, n_large_crops, large_size, n_small_crops, small_size, strong=True)` wraps any torchvision-style dataset that returns `(PIL_Image, label)` tuples. It applies two `ContrastiveAugmentation` instances internally: one at `large_size` and one at `small_size`. `__getitem__` returns `(crops_list, label)` where `crops_list` is a flat Python list of `n_large_crops + n_small_crops` tensors.

### ssl_collate_multi_crop

Collates a batch of `(crops_list, label)` tuples into `(list_of_stacked_tensors, labels_tensor)`. Returns a **list** of `n_crops` tensors (not a single stacked tensor) because large crops are `[B,C,224,224]` and small crops are `[B,C,96,96]` -- they cannot be stacked together.

### SSLDataModule updates

- Added optional `dataset` parameter to `__init__`. When provided, `setup()` sets `self.train_dataset = self.dataset` directly, skipping `ImageFolder` creation.
- `train_dataloader()` now uses `isinstance(self.train_dataset, MultiCropDataset)` to select `ssl_collate_multi_crop` vs `ssl_collate_fn`.

## Tests

6 tests in `tests/test_multi_crop.py`:

| Test | Behavior Verified |
|------|-------------------|
| `test_returns_8_crops_and_label` | 2+6 config yields 8-element list |
| `test_large_crops_are_224_small_crops_are_96` | Spatial dimensions correct |
| `test_ssl_collate_multi_crop_shapes` | Collated batch shapes: [B,C,224,224] x2, [B,C,96,96] x6 |
| `test_edge_case_zero_small_crops` | n_small_crops=0 yields exactly 2 crops |
| `test_preserves_labels_from_base_dataset` | Labels pass through unchanged |
| `test_ssldatamodule_uses_multi_crop_collate` | SSLDataModule selects correct collate_fn |

Full test suite: **147/147 passing** (141 prior + 6 new).

## Commits

| Hash | Message |
|------|---------|
| fa7ed0e | feat(05-01): implement MultiCropDataset, ssl_collate_multi_crop, SSLDataModule multi-crop support |

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None. MultiCropDataset is an internal training-time wrapper with no network endpoints or new trust boundaries.

## Self-Check: PASSED

- `core/data.py` contains `class MultiCropDataset(torch.utils.data.Dataset):` -- FOUND
- `core/data.py` contains `def ssl_collate_multi_crop(batch):` -- FOUND
- `tests/test_multi_crop.py` exists -- FOUND
- Commit `fa7ed0e` exists -- FOUND
- All 147 tests pass -- VERIFIED
