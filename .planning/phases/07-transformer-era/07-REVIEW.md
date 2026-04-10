---
phase: 07-transformer-era
reviewed: 2026-04-10T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - configs/dino_vit_small.yaml
  - configs/moco_v3_vit_small.yaml
  - eval/__init__.py
  - eval/dinov2_demo.py
  - methods/__init__.py
  - methods/dino/__init__.py
  - methods/dino/module.py
  - methods/moco_v3/__init__.py
  - methods/moco_v3/module.py
  - tests/test_dino.py
  - tests/test_dinov2_demo.py
  - tests/test_moco_v3.py
  - tests/test_smoke_transformer.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 07: Code Review Report

**Reviewed:** 2026-04-10T00:00:00Z
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

This phase adds MoCo v3 and DINO transformer-era SSL methods, a DINOv2 feature extraction demo, and a full suite of unit and smoke tests. The core algorithmic logic is sound: centering-before-loss ordering is correctly implemented in DINO, the patch projection freeze is correctly applied before the momentum encoder deepcopy in MoCo v3, and EMA updates are properly deferred to `on_train_batch_end`. No critical security or data-loss issues were found.

Four warnings require attention: a dead conditional branch in `MoCoV3Module.training_step` that silently masks structural issues, a silent directory fallback in the DINOv2 demo that contradicts its own docstring, a missing `None`-guard before accessing `patch_embed.proj.bias`, and an inverted description in `_get_teacher_temp`'s docstring that misrepresents the warmup direction. Three info-level items cover repeated `import itertools` inside a property, inconsistent NaN-check idioms across test files, and an orphan `swav:` block embedded in the DINO config.

---

## Warnings

### WR-01: Dead conditional branch in `MoCoV3Module.training_step` — both arms are identical

**File:** `methods/moco_v3/module.py:192-195`
**Issue:** The `if isinstance(views, torch.Tensor)` block and the `else` block both execute `v1, v2 = views[0], views[1]`. The branching adds no behaviour — it silently accepts both a stacked tensor `[2, B, C, H, W]` and a list/tuple, but produces the same result either way. For a stacked tensor, `views[0]` and `views[1]` do correctly slice along dim 0, so the code works, but the conditional is dead and misleads readers into thinking different handling occurs. If the intent was to handle stacked tensors differently (e.g., `v1 = views[0]` vs `v1, v2 = views.unbind(0)`), the current form masks an unimplemented path.

**Fix:** Remove the dead branch and use a single unpacking that works for both the list and tensor cases:
```python
# Both list/tuple and stacked tensor [2, B, C, H, W] support indexing
v1, v2 = views[0], views[1]
```
If stacked-tensor support is genuinely intended as a separate path, implement distinct logic (e.g., `views.unbind(0)` for tensors vs direct indexing for sequences):
```python
if isinstance(views, torch.Tensor):
    v1, v2 = views.unbind(0)  # [2,B,C,H,W] -> two [B,C,H,W] tensors
else:
    v1, v2 = views[0], views[1]
```

---

### WR-02: Silent directory fallback in `load_datasets` contradicts docstring contract

**File:** `eval/dinov2_demo.py:117-122`
**Issue:** When `--dataset imagefolder` is used, if `train_path` (`data_dir/train`) is not a directory, the code silently reassigns `train_path = args.data_dir`. Likewise for `test_path`. The docstring states the function raises `ValueError` if `--data-dir` doesn't exist, but no such check is made. In practice both `train_path` and `test_path` can end up pointing to the same directory (`args.data_dir`), training and evaluating on identical data without any warning. If `args.data_dir` itself does not contain a valid `ImageFolder` layout, `ImageFolder` raises an `OSError` rather than the promised `ValueError`, breaking the documented contract.

**Fix:** Raise `ValueError` explicitly when a required path is absent, rather than falling back silently:
```python
elif args.dataset == "imagefolder":
    import os
    if not os.path.isdir(args.data_dir):
        raise ValueError(
            f"--data-dir {args.data_dir!r} does not exist or is not a directory"
        )
    train_path = os.path.join(args.data_dir, "train")
    test_path = os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_path):
        raise ValueError(
            f"ImageFolder expects a 'train/' subdirectory under {args.data_dir!r}, "
            f"but {train_path!r} was not found."
        )
    if not os.path.isdir(test_path):
        raise ValueError(
            f"ImageFolder expects a 'val/' subdirectory under {args.data_dir!r}, "
            f"but {test_path!r} was not found."
        )
    train_ds = ImageFolder(root=train_path, transform=transform)
    test_ds = ImageFolder(root=test_path, transform=transform)
```

---

### WR-03: `patch_embed.proj.bias` accessed without checking for `None`

**File:** `methods/moco_v3/module.py:117`
**Issue:** Line 117 calls `.requires_grad_(False)` on `self.backbone.patch_embed.proj.bias` without first checking whether the bias exists. The standard timm `vit_small_patch16_224` always includes a bias on the patch embedding Conv2d, so this does not fail in practice, but it is fragile: any ViT variant constructed with `bias=False` (e.g., a custom backbone, an updated timm version, or a call to `build_backbone` with such a model) will raise `AttributeError: 'NoneType' object has no attribute 'requires_grad_'` at module construction time.

**Fix:** Guard the bias freeze with a `None` check:
```python
if hasattr(self.backbone, "patch_embed"):
    self.backbone.patch_embed.proj.weight.requires_grad_(False)
    if self.backbone.patch_embed.proj.bias is not None:
        self.backbone.patch_embed.proj.bias.requires_grad_(False)
```

---

### WR-04: `_get_teacher_temp` docstring describes warmup direction backwards

**File:** `methods/dino/module.py:143-155`
**Issue:** The docstring states "Warms up from `teacher_temp` (0.04) to `warmup_teacher_temp` (0.07)" but this is the wrong direction. The actual warmup starts at `warmup_teacher_temp` (0.07, high/soft) and ends at `teacher_temp` (0.04, low/sharp) — the code on line 154 is correct. The docstring on line 143 and the "Note" on line 148 are inverted relative to each other: line 143 says the range is "from teacher_temp to warmup_teacher_temp" but line 148 says "start at warmup_teacher_temp (0.07), end at teacher_temp (0.04)". This inconsistency risks a maintainer reversing the interpolation when the code is next modified.

**Fix:** Correct the opening description to match the note and the code:
```python
def _get_teacher_temp(self) -> float:
    """Compute teacher temperature with linear warmup.

    Warms up from ``warmup_teacher_temp`` (0.07, high/soft) down to
    ``teacher_temp`` (0.04, low/sharp) over ``warmup_teacher_temp_epochs``
    epochs, then holds at ``teacher_temp``.

    Note: High teacher temperature early produces softer distributions that
    stabilize initial training; the temperature decreases (sharpens) as
    training progresses per the DINO paper.
    """
```

---

## Info

### IN-01: `import itertools` inside a property called every training step

**File:** `methods/dino/module.py:132`, `methods/moco_v3/module.py:172`
**Issue:** Both `learnable_params` properties import `itertools` inline on every invocation. While Python caches module imports, accessing `sys.modules` on each step adds unnecessary overhead at the critical training-step frequency. It also obscures the module's dependencies from readers who scan the top-level imports.

**Fix:** Move `import itertools` to the top of each module file:
```python
import itertools  # add at module level alongside other stdlib imports
```

---

### IN-02: Inconsistent NaN-check idiom between test files

**File:** `tests/test_dino.py:406`
**Issue:** `test_dino.py` line 406 checks for NaN with `assert not (loss != loss)`, relying on the IEEE 754 property that `NaN != NaN`. `test_smoke_transformer.py` uses `math.isfinite()` and `test_moco_v3.py` uses `assert loss == loss`. Three different idioms for the same check reduces readability. `math.isnan()` or `torch.isnan()` is the idiomatic Python/PyTorch choice and makes the intent explicit.

**Fix:** Standardise across all test files:
```python
import math
assert not math.isnan(loss), f"Loss at epoch {i} is NaN"
assert math.isfinite(loss), f"Loss at epoch {i} is infinite"
```

---

### IN-03: Orphan `swav:` config block embedded in `dino_vit_small.yaml`

**File:** `configs/dino_vit_small.yaml:40-49`
**Issue:** The DINO config file contains a full `swav:` sub-block (lines 40-49). The accompanying comment says it "documents multi-crop settings for reference" and notes the data pipeline is configured programmatically. However, if `TrainConfig` defines a `swav` field (as it likely does to support the SwAV method), this block will silently populate `cfg.swav` with values during `load_config`. This is benign only as long as the DINO training path never reads `cfg.swav`; if it ever does (e.g., through shared multi-crop infrastructure), the embedded values will take effect silently, overriding intended defaults.

**Fix:** Remove the `swav:` block from the DINO config. If multi-crop size reference values are needed for documentation, move them to comments:
```yaml
# Multi-crop: 2 global (224x224) + 6 local (96x96) = 8 views total.
# These are configured programmatically in train.py via MultiCropDataset,
# not via YAML. See configs/swav_vit_small.yaml for the canonical SwAV config.
```

---

_Reviewed: 2026-04-10T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
