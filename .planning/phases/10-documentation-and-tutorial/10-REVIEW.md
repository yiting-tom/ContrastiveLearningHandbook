---
phase: 10-documentation-and-tutorial
reviewed: 2026-05-03T00:00:00Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - README.md
  - docs/tutorial.md
  - docs/tutorial/01_add_a_new_method.md
  - docs/tutorial/02_running_an_experiment.md
  - docs/tutorial/03_comparing_methods.md
  - methods/barlow_twins/module.py
  - methods/byol/module.py
  - methods/dino/module.py
  - methods/infomin/module.py
  - methods/simsiam/module.py
  - methods/supcon/module.py
  - methods/swav/module.py
  - tests/test_docstrings.py
  - tests/test_train_script.py
  - train.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-05-03T00:00:00Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Phase 10 delivers documentation, tutorials, and DOC-02 compliance docstrings across all 14 SSL method modules. The tutorial content (README, docs/tutorial.md, and the three per-section docs) is clear, accurate, and internally consistent. The docstring coverage in every reviewed method module is thorough. The test suite for docstring compliance and the train.py smoke tests are well-structured.

One critical security issue exists in `SupConFinetuneModule.from_stage1_ckpt`: `torch.load` is called without `weights_only=True`, allowing arbitrary code execution from a malicious checkpoint file. Four warnings cover a documentation-code mismatch in Barlow Twins (L2-norm vs per-dimension standardization), a naming inversion in DINO's teacher temperature warmup that makes the docstring contradict the code, a grayscale-probability discrepancy in InfoMin between the class docstring and the module docstring, and a `build_projector` signature deviation in SupConModule that violates the interface contract documented in the tutorial. Three info items cover minor dead imports, a torch.load deprecation warning risk, and a commented-out note that could mislead readers.

---

## Critical Issues

### CR-01: Unsafe `torch.load` in `from_stage1_ckpt` — arbitrary code execution risk

**File:** `methods/supcon/module.py:290`
**Issue:** `torch.load(ckpt_path, map_location=map_location)` is called without `weights_only=True`. Since PyTorch 2.0, loading a checkpoint without `weights_only=True` uses Python pickle without restriction, allowing a crafted checkpoint file to execute arbitrary code on the machine running the script. This is a documented CVE-class risk that PyTorch itself warns about at runtime (a `FutureWarning` in 2.x, an error in future versions).
**Fix:**
```python
state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
```
If the checkpoint contains non-tensor objects that require full pickle (e.g., Lightning's `hyper_parameters`), use the safe allowlist pattern instead:
```python
state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
# ... and document the trusted-source assumption in a comment
```
The tutorial teaches users to pass checkpoint paths as CLI arguments; this path must not silently deserialize untrusted data.

---

## Warnings

### WR-01: Barlow Twins applies L2-norm per sample, but paper requires per-dimension standardization

**File:** `methods/barlow_twins/module.py:104-105`
**Issue:** The code applies `F.normalize(self.projector(...), dim=1)` (L2-normalizes each sample vector) before computing the cross-correlation matrix. The Barlow Twins paper (and the class docstring algorithm step 3) explicitly says "Standardize (zero mean, unit variance) projector outputs **across batch**", meaning per-dimension z-scoring, not per-sample L2-norm. These two operations are mathematically distinct. Per-sample L2-norm does not make each feature dimension zero-mean and unit-variance across the batch. The module-level gotcha "L2-normalize the projector outputs" reinforces the wrong operation. The reference implementation (`facebookresearch/barlowtwins`) uses batch normalization at the projector output to achieve the per-dimension standardization described in the paper.
**Fix:** Replace `F.normalize(..., dim=1)` with batch normalization of the projector output, or explicitly z-score across the batch dimension:
```python
# Option A: use the ProjectionHead's built-in BN at the output layer (already present)
# and remove F.normalize — the BN already standardizes per-dimension across the batch.
z_a = self.projector(self.backbone(v1))  # ProjectionHead BN handles standardization
z_b = self.projector(self.backbone(v2))

# Option B (explicit): standardize manually
def _standardize(z):
    return (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-5)
z_a = _standardize(self.projector(self.backbone(v1)))
z_b = _standardize(self.projector(self.backbone(v2)))
```
Also correct the module-level gotcha to say "Standardize per-dimension across the batch" rather than "L2-normalize".

### WR-02: DINO `_get_teacher_temp` docstring inverts warmup direction relative to code

**File:** `methods/dino/module.py:162-176`
**Issue:** The docstring says "Warms up from `teacher_temp` (0.04) to `warmup_teacher_temp` (0.07)" but the code interpolates in the opposite direction: `warmup_teacher_temp + alpha * (teacher_temp - warmup_teacher_temp)`. At `alpha=0` (epoch 0) the return value is `warmup_teacher_temp` (0.07); at `alpha=1` (end of warmup) it returns `teacher_temp` (0.04). The temperature therefore **cools** from 0.07 to 0.04, not warms. The in-code note "The warmup starts HIGH (0.07) and ends LOW (0.04)" correctly describes the behavior, but directly contradicts the first sentence of the docstring. A reader following the docstring summary will have the direction backwards.
**Fix:** Rewrite the docstring first sentence to match the code:
```python
"""Compute teacher temperature with linear cooldown during warmup phase.

Starts at ``warmup_teacher_temp`` (0.07, high = soft distributions) at epoch 0
and linearly decreases to ``teacher_temp`` (0.04, low = sharp distributions)
over ``warmup_teacher_temp_epochs`` epochs, then stays at ``teacher_temp``.

High temperature early = softer distributions = more stable initial training
(per DINO paper Section 3.2).
"""
```

### WR-03: InfoMin class docstring states grayscale_prob=0.5, but default is 0.4 everywhere else

**File:** `methods/infomin/module.py:62`
**Issue:** The class-level docstring (line 62) says "aggressive color jitter, random grayscale p=0.5, no Gaussian blur" as the algorithm step. The module-level docstring (line 19) and `build_augmentation` signature (line 88) both use `grayscale_prob=0.4` as the documented and default value. The `InfoMinConfig` default (not visible here, in `core/config.py`) also defaults to 0.4 per the module docstring. The class docstring value 0.5 appears nowhere else and is inconsistent.
**Fix:** Update the class docstring algorithm description to match:
```python
    Algorithm:
    1. Wrap a SimCLR-style backbone with the "InfoMin" augmentation policy
       (aggressive color jitter s=1.5, random grayscale p=0.4, no Gaussian blur).
```

### WR-04: `SupConModule.build_projector` deviates from the `BaseSSLModule` interface contract documented in the tutorial

**File:** `methods/supcon/module.py:90`
**Issue:** The tutorial (both `docs/tutorial.md` and `docs/tutorial/01_add_a_new_method.md`) declares the mandatory interface as `build_projector(self) -> nn.Module`. `SupConModule` overrides it as `def build_projector(self, supcon_cfg: SupConConfig | None = None) -> nn.Module`. This works in Python because the extra parameter has a default, but it breaks the documented interface contract. A learner following the tutorial to understand how existing methods work will see a method that does not match the stated pattern, and a learner subclassing `SupConModule` who calls `super().build_projector()` will trigger the extra parameter path without passing a config, relying on the `None` default which then re-fetches `self.cfg`. The extra parameter path is also inconsistent: `build_projector()` with no args re-fetches cfg from `self.cfg`, but the call site at line 86 passes `supcon_cfg` explicitly — meaning the method exists in two modes.
**Fix:** Remove the optional parameter and always use `self.cfg` internally:
```python
def build_projector(self) -> nn.Module:
    """2-layer MLP projection head (same architecture as SimCLR v1)."""
    supcon_cfg = self.cfg.supcon or SupConConfig()
    return ProjectionHead(
        self.feat_dim,
        hidden_dim=2048,
        output_dim=supcon_cfg.projection_dim,
        num_layers=2,
    )
```
Then update the `__init__` call site from `self.build_projector(supcon_cfg)` to `self.build_projector()`.

---

## Info

### IN-01: `import itertools` inside method body in two modules

**File:** `methods/dino/module.py:154`, `methods/swav/module.py:115`
**Issue:** Both `DINOModule.learnable_params` and `SwAVModule.learnable_params` have `import itertools` inside the property body. This is a deferred import pattern that works but is unusual — `itertools` is a standard library module that has no import cost. Placing imports at the top-level follows Python convention and makes the dependency explicit.
**Fix:** Move `import itertools` to the top of each file alongside the other standard-library imports.

### IN-02: `test_train_script.py` — `sys.modules` cleanup only removes `"train"`, not re-imported submodules

**File:** `tests/test_train_script.py:91-92`
**Issue:** The test removes `sys.modules["train"]` and then `import train` to get a fresh module. However, `train.py` imports `methods` (which triggers side-effect dispatcher registrations) and Lightning callbacks. If the `train` module import fails partway through in a prior test run, leftover partial state in `sys.modules` from transitively imported modules may cause the second test (`test_train_py_invalid_config_raises`) to behave differently than a fresh process. The pattern is fragile but not currently broken. The same cleanup block appears in two tests (lines 91-92 and 103-104) as copy-paste.
**Fix:** Extract into a helper or use `importlib.reload`:
```python
import importlib, sys

def _fresh_train_module():
    sys.modules.pop("train", None)
    import train as _train
    return importlib.reload(_train)
```

### IN-03: README quickstart checkpoint path example (`checkpoints/last.ckpt`) does not match Lightning's actual output pattern

**File:** `README.md:43`
**Issue:** The quickstart step 4 shows `--ckpt checkpoints/last.ckpt`. Lightning does not write `checkpoints/last.ckpt` by default — it writes `lightning_logs/version_<N>/checkpoints/epoch=<E>-step=<S>.ckpt`. The correct path format is documented correctly in `docs/tutorial/02_running_an_experiment.md` (step 4). The README quickstart uses a simplified path that will not work without additional trainer configuration (e.g., `ModelCheckpoint(filename="last")`).
**Fix:** Update the README quickstart to use the actual Lightning default path pattern or add a note that the path must be substituted:
```bash
# 4. Linear probe evaluation (substitute your actual checkpoint path)
CKPT=$(ls lightning_logs/version_0/checkpoints/*.ckpt | tail -1)
python eval/linear_probe.py configs/simclr_v1_resnet18.yaml --ckpt "$CKPT"
```

---

_Reviewed: 2026-05-03T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
