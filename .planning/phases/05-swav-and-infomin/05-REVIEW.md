---
phase: 05-swav-and-infomin
reviewed: 2026-04-08T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - configs/infomin_resnet18.yaml
  - configs/swav_resnet18.yaml
  - core/config.py
  - core/data.py
  - methods/__init__.py
  - methods/infomin/__init__.py
  - methods/infomin/module.py
  - methods/simclr/module.py
  - methods/swav/__init__.py
  - methods/swav/losses.py
  - methods/swav/module.py
  - methods/swav/prototype.py
  - tests/test_config.py
  - tests/test_infomin.py
  - tests/test_multi_crop.py
  - tests/test_swav.py
  - tools/compare_augmentations.py
findings:
  critical: 2
  warning: 3
  info: 3
  total: 8
status: issues_found
---

# Phase 05: Code Review Report

**Reviewed:** 2026-04-08
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

This phase adds SwAV (multi-crop + Sinkhorn-Knopp optimal transport) and InfoMin
(augmentation-policy variant of SimCLR) to the tutorial codebase. The SwAV
implementation is well-structured: the Sinkhorn-Knopp function is numerically
stable (log-sum-exp trick applied), the prototype layer correctly handles L2
normalization and gradient freezing, and the test suite is comprehensive.

Two critical bugs exist in `core/config.py`: `InfoMinConfig` is defined twice,
and the `infomin` field is declared twice in `TrainConfig`. Python resolves both
silently by using the last definition, but the first `InfoMinConfig` definition
(which includes `temperature` and `projection_dim`) is completely shadowed and
unreachable. This is a correctness bug — the `temperature` field intended for
InfoMin YAML overrides is silently dropped.

Three warnings cover a potential division-by-zero in `swav_loss`, a missing
`super().setup()` call in `InfoMinModule`, and a fragile test assertion pattern.

---

## Critical Issues

### CR-01: `InfoMinConfig` defined twice in `core/config.py` — first definition silently lost

**File:** `core/config.py:72-83` and `core/config.py:130-140`

**Issue:** `InfoMinConfig` is defined twice. Python applies the second definition
at module load, completely discarding the first. The first definition (lines 72–83)
includes `temperature: float = 0.5` and `projection_dim: int = 128`, enabling
YAML-controlled overrides of loss temperature and projection dimension for InfoMin.
The second definition (lines 130–140) omits both fields. Any code written to access
`cfg.infomin.temperature` will raise `AttributeError` at runtime because the
effective `InfoMinConfig` class has no `temperature` attribute.

Additionally, `test_config.py:188` (`test_infomin_config_defaults`) only checks
`color_strength`, `grayscale_prob`, and `use_blur`, so the tests do not catch this
regression.

**Fix:** Delete the first `InfoMinConfig` definition (lines 72–83). If `temperature`
and `projection_dim` are intentional for the InfoMin config, add them to the second
(retained) definition instead:

```python
# core/config.py — single InfoMinConfig definition (lines 130-140, keep this one)
class InfoMinConfig(_StrictBase):
    """InfoMin (Tian et al., NeurIPS 2020) method-specific hyper-parameters."""

    color_strength: float = 1.5
    grayscale_prob: float = 0.4
    use_blur: bool = False
    # Add back only if InfoMin needs its own temperature/projection_dim:
    # temperature: float = 0.5
    # projection_dim: int = 128
```

Remove lines 72–83 entirely (the first `InfoMinConfig` class definition).

---

### CR-02: `infomin` field declared twice in `TrainConfig` — silent Pydantic field shadowing

**File:** `core/config.py:243` and `core/config.py:250`

**Issue:** `TrainConfig` declares `infomin: Optional[InfoMinConfig] = None` twice
(lines 243 and 250). Pydantic v2 processes field declarations in order; the second
declaration overwrites the first. While the runtime behavior is currently identical
(both refer to the same class), this constitutes a hidden naming error that will
silently break if either declaration is changed independently. It also causes
confusion since the `extra='forbid'` schema is expected to be explicit and clean.

**Fix:** Remove the duplicate field declaration at line 250:

```python
# core/config.py — TrainConfig field declarations
# Keep only ONE of these two lines:
infomin: Optional[InfoMinConfig] = None   # line 243 — keep
# infomin: Optional[InfoMinConfig] = None  # line 250 — DELETE this one
```

---

## Warnings

### WR-01: `swav_loss` divides by zero when `n_crops == 1`

**File:** `methods/swav/losses.py:140`

**Issue:** The loss normalization divides by `n_large_crops * (n_crops - 1)`. If
`n_crops` equals `n_large_crops` (e.g., both are 1), `(n_crops - 1)` is 0 and the
division produces `inf` or `nan`. With `n_crops == 1` and `n_large_crops == 1`,
the inner loop body (`for v in range(n_crops): if v == i: continue`) executes zero
times, so `loss` remains 0.0, and `0.0 / 0` yields `nan` on some backends.
Even with `n_large_crops=2, n_small_crops=0` (n_crops=2), the formula gives
`2 * 1 = 2` which is fine; the edge case is `n_crops=1`.

**Fix:** Guard against degenerate input or add an assertion:

```python
# methods/swav/losses.py
n_pairs = n_large_crops * (n_crops - 1)
if n_pairs == 0:
    raise ValueError(
        f"swav_loss requires at least 2 crops total (got n_crops={n_crops}). "
        "Set n_small_crops >= 1 or use n_large_crops >= 2."
    )
loss /= n_pairs
```

---

### WR-02: `InfoMinModule.setup()` does not call `super().setup()`

**File:** `methods/infomin/module.py:97-119`

**Issue:** `InfoMinModule.setup()` overrides `setup(stage=None)` to wire the
InfoMin augmentation into the data pipeline. It does not call `super().setup(stage)`.
`SimCLRv1Module` does not define `setup()`, so the call skips up to `BaseSSLModule`
(and ultimately `LightningModule.setup()`). If `BaseSSLModule` adds setup logic in a
future phase (e.g., to initialize evaluation metrics, checkpointing hooks, or
distributed state), `InfoMinModule` will silently skip it. Additionally, the current
`setup()` does not set `self.val_dataset`, while `SSLDataModule.setup()` does set it
conditionally. If any eval code expects `val_dataset` on the module, this will raise
`AttributeError`.

**Fix:**

```python
# methods/infomin/module.py
def setup(self, stage=None):
    super().setup(stage)  # Always call parent first
    import os
    from torchvision.datasets import ImageFolder
    ...
```

---

### WR-03: NaN test uses `loss == loss` identity pattern (fragile and non-obvious)

**File:** `tests/test_infomin.py:307`, `tests/test_infomin.py:376`, `tests/test_swav.py:335`

**Issue:** NaN detection is written as `assert loss == loss`, exploiting the IEEE 754
property that `NaN != NaN`. This pattern is not self-documenting and can be
misread as a tautology (always true). A reader unfamiliar with this idiom may remove
the assertion thinking it is dead code.

**Fix:** Use an explicit check:

```python
# Replace this pattern:
assert loss == loss, f"Epoch {i} loss is NaN"

# With:
assert not (loss != loss), f"Epoch {i} loss is NaN"
# Or, cleaner:
import math
assert math.isfinite(loss), f"Epoch {i} loss is NaN or infinite: {loss}"
```

---

## Info

### IN-01: `denormalize()` return type annotation references `np.ndarray` without top-level numpy import

**File:** `tools/compare_augmentations.py:36`

**Issue:** The function signature is `def denormalize(tensor: torch.Tensor) -> "np.ndarray":`.
`numpy` is imported inside the function body (`import numpy as np` at line 39), not
at module level. The string annotation `"np.ndarray"` is never resolved at import
time (forward-reference string form), so this does not cause a runtime error, but
any static type checker or documentation tool that resolves annotations will fail to
resolve `np.ndarray`. The `numpy` import should be at the module level for clarity.

**Fix:**

```python
# tools/compare_augmentations.py — add at top of imports
import numpy as np

# Change signature to:
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.clone().float()
    img = img * IMAGENET_STD_T + IMAGENET_MEAN_T
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
    # Remove the `import numpy as np` line that was inside the function body
```

---

### IN-02: `InfoMinConfig` (first definition, lines 72-83) includes `temperature` and `projection_dim` that are absent from the effective second definition

**File:** `core/config.py:79-80`

**Issue:** Related to CR-01. The first `InfoMinConfig` definition adds
`temperature: float = 0.5` and `projection_dim: int = 128`, suggesting the original
intent was to allow per-YAML tuning of these parameters for InfoMin. The second
definition (which Python uses) has neither. If the InfoMin paper's temperature
differs from SimCLR's default (0.5), users cannot override it via YAML. The
`infomin_resnet18.yaml` config also has no `temperature` key, consistent with the
second definition — but this means the InfoMin loss temperature is always locked to
SimCLR's default.

**Fix:** After resolving CR-01, deliberately decide whether InfoMin needs its own
`temperature` field. If yes, add it to the single retained definition and to
`configs/infomin_resnet18.yaml`. If no, remove it entirely and document that InfoMin
inherits SimCLR's `temperature` via the `simclr:` sub-config block.

---

### IN-03: Magic number `128` hardcoded in `SwAVModule.__init__` for prototype input dim

**File:** `methods/swav/module.py:62`

**Issue:** `PrototypeLayer(feat_dim=128, ...)` hardcodes 128 as the projection output
dimension. This is consistent with `build_projector()` which also hardcodes 128, but
the two sites are not linked. If `build_projector` is overridden in a subclass to
produce a different output dimension, `PrototypeLayer` will receive features of the
wrong dimension and silently produce incorrect scores (no shape-mismatch error at
construction time, only at forward time when `nn.Linear` weight shapes mismatch).

**Fix:** Extract the projection output dimension as a class-level constant or derive
it from the projector:

```python
# methods/swav/module.py
PROJ_DIM = 128  # class constant or config field

self.projector = self.build_projector()
self.prototype_layer = PrototypeLayer(
    feat_dim=PROJ_DIM,
    n_prototypes=swav_cfg.n_prototypes,
)
```

Or, derive `feat_dim` dynamically by inspecting the projector's output layer after
construction (safer for subclassing).

---

_Reviewed: 2026-04-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
