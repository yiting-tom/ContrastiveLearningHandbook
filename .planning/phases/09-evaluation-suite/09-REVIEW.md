---
phase: 09-evaluation-suite
reviewed: 2026-04-11T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - eval/__init__.py
  - eval/cam_vis.py
  - eval/finetune.py
  - eval/knn_callback.py
  - eval/linear_probe.py
  - eval/tsne_vis.py
  - eval/umap_vis.py
  - pyproject.toml
  - requirements.txt
  - tests/conftest.py
  - tests/test_eval_cam.py
  - tests/test_eval_finetune.py
  - tests/test_eval_integration.py
  - tests/test_eval_knn.py
  - tests/test_eval_linear_probe.py
  - tests/test_eval_tsne.py
  - tests/test_eval_umap.py
findings:
  critical: 0
  warning: 5
  info: 6
  total: 11
status: issues_found
---

# Phase 09: Code Review Report

**Reviewed:** 2026-04-11T00:00:00Z
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

This phase delivers a six-component evaluation suite (k-NN callback, linear probe, fine-tuning, t-SNE, UMAP, and CAM visualization) plus a comprehensive test suite covering unit and integration scenarios. The overall implementation is solid: error paths are guarded, cache keys are checkpoint-scoped, multi-view batches are handled consistently, and the FAISS/brute-force split in knn_predict is cleanly factored. No critical (security, crash, data-loss) issues were found.

Five warnings flag logic gaps that could cause silent incorrect results or crashes at runtime: an unchecked empty-dataloader edge case in the k-NN callback, an `assert` used for a runtime invariant inside an optimizer configuration path, incorrect num_classes inference from a single-column label set, a data-type assumption on the FAISS CPU path that will crash on GPU tensors, and an unclosed file handle in two CLI main() functions. Six informational items cover minor quality concerns.

---

## Warnings

### WR-01: `num_classes` inferred from `max(label) + 1` — wrong when classes are not contiguous

**File:** `eval/knn_callback.py:227`
**Issue:** `num_classes = int(train_labels.max().item()) + 1` is only correct when labels are 0-indexed and contiguous. If a split happens to miss a class entirely (e.g., a very small dataset where a class has no training examples), `num_classes` will be under-counted, causing `votes` to have fewer columns than actual classes and `votes.scatter_add_` to raise an index-out-of-range error or silently misclassify. The same pattern appears in `eval/linear_probe.py:243` and `tests/test_eval_integration.py:448`.
**Fix:**
```python
# Prefer the datamodule's known class count when available, or use unique labels
num_classes = len(train_labels.unique())
# Or, if the datamodule exposes it:
# num_classes = len(trainer.datamodule.train_dataset.classes)
```

### WR-02: `assert` statement used as a runtime invariant guard in `configure_optimizers`

**File:** `eval/linear_probe.py:145`
**Issue:** `assert optimizer.param_groups[0]["weight_decay"] == 0.0` is a runtime check that will be silently disabled if Python runs with the `-O` (optimize) flag (`python -O`). If a future refactor accidentally passes a non-zero `weight_decay`, the error will be swallowed in optimized mode and the probe will train with regularization, producing wrong evaluation numbers with no visible error.
**Fix:**
```python
wd = optimizer.param_groups[0]["weight_decay"]
if wd != 0.0:
    raise ValueError(
        f"Linear probe requires weight_decay=0.0, got {wd}. "
        "Regularizing the only learnable layer degrades probe accuracy."
    )
```

### WR-03: FAISS path calls `.numpy()` on tensors that may not be on CPU

**File:** `eval/knn_callback.py:83-88`
**Issue:** In the FAISS branch (`n_train > FAISS_THRESHOLD`), the code calls `train_features.numpy()` and `test_features.numpy()` without checking the device. In `_extract_features`, features are moved to CPU before returning, so this is safe in the callback path. However, `knn_predict` is also a public API imported by `test_eval_integration.py` and by users who may pass GPU tensors directly. Calling `.numpy()` on a CUDA tensor raises a `RuntimeError` immediately.
**Fix:**
```python
index.add(
    np.ascontiguousarray(train_features.cpu().numpy().astype(np.float32))
)
raw_sims, raw_idx = index.search(
    np.ascontiguousarray(test_features.cpu().numpy().astype(np.float32)),
    k,
)
```

### WR-04: `on_validation_epoch_end` calls `train_dataloader()` unconditionally before checking size

**File:** `eval/knn_callback.py:221-222`
**Issue:** After the `val_loader is None` guard, the callback always calls `trainer.datamodule.train_dataloader()`. If the train dataloader returns an empty iterator (e.g., a pathological empty dataset in tests or edge-case configs), `_extract_features` will produce an empty `all_features` list, and `torch.cat([])` at line 190 will raise a `RuntimeError: expected a non-empty list of Tensors`. This edge case is not guarded.
**Fix:**
```python
train_feats, train_labels = self._extract_features(pl_module, train_loader)
if train_feats.shape[0] == 0:
    return  # nothing to build the feature bank from
```

### WR-05: `open(args.config)` used without `with` statement in `tsne_vis.py` and `umap_vis.py` CLIs

**File:** `eval/tsne_vis.py:207`, `eval/umap_vis.py:194`
**Issue:** Both `main()` functions use `yaml.safe_load(open(args.config))` as a bare expression, which leaves the file handle open until the garbage collector closes it. This is a resource leak; on some systems it can prevent the process from opening additional files under a low file-descriptor limit. Identical pattern in both files.
**Fix:**
```python
with open(args.config) as f:
    cfg = TrainConfig.model_validate(yaml.safe_load(f))
```
(The pattern is already used correctly in `eval/finetune.py:179` and `eval/linear_probe.py:203`.)

---

## Info

### IN-01: `sklearn` not listed in `requirements.txt`

**File:** `requirements.txt`
**Issue:** `eval/tsne_vis.py` imports `from sklearn.decomposition import PCA` and `from sklearn.manifold import TSNE`. `scikit-learn` is not listed in `requirements.txt` or `pyproject.toml`. This will cause an `ImportError` for anyone who installs only from the listed requirements.
**Fix:** Add `scikit-learn>=1.3` to `requirements.txt`.

### IN-02: `matplotlib` not listed in `requirements.txt`

**File:** `requirements.txt`
**Issue:** `eval/cam_vis.py`, `eval/tsne_vis.py`, and `eval/umap_vis.py` all `import matplotlib`. It is not listed in `requirements.txt` or `pyproject.toml`.
**Fix:** Add `matplotlib>=3.7` to `requirements.txt`.

### IN-03: Duplicated `extract_features` function across `tsne_vis.py` and `umap_vis.py`

**File:** `eval/tsne_vis.py:30-79`, `eval/umap_vis.py:25-74`
**Issue:** Both files contain byte-for-byte identical `extract_features` functions (including docstring, logic, and multi-view handling). Any fix in one will need to be manually mirrored to the other. This is maintenance-prone duplication.
**Fix:** Extract the shared function into `eval/_utils.py` (or add it to `eval/__init__.py`) and import it from both modules.

### IN-04: `TestRunCAM` test has dead code and misleading variable rebinding

**File:** `tests/test_eval_cam.py:271-285`
**Issue:** The test creates `MockModel` and sets `backbone, feat_dim = None, None`, then immediately overwrites both with a `try/except timm` block. The `MockModel` class defined at lines 259-270 is never instantiated. The `feat_dim` variable is never used after the rebind. The dead `MockModel` and `feat_dim` create reader confusion about what the test actually does.
**Fix:** Remove the unused `MockModel` class, the unused `backbone = FakeResNetBackbone()` line (line 262), and the stale `feat_dim = None` assignment.

### IN-05: Magic number `256` as DataLoader batch size in `linear_probe.py`

**File:** `eval/linear_probe.py:252-253`
**Issue:** The cached-feature DataLoaders are created with a hardcoded `batch_size=256` not taken from `lp_cfg` or any config field. If users configure a small memory budget, this value cannot be changed without editing source code.
**Fix:** Either expose `batch_size` in `LinearProbeConfig` (with a default of 256) or at minimum define it as a named constant at the top of the file.

### IN-06: `pyproject.toml` missing `[project]` or `[build-system]` table — not a valid package descriptor

**File:** `pyproject.toml`
**Issue:** The file contains only `[tool.pytest.ini_options]`. Without a `[project]` section, `pip install .` will fail with a `build-system` error. If the project is intended to be installable (e.g., for CI), the toml is incomplete. If it is only used for pytest configuration, a plain `pytest.ini` or `setup.cfg` `[tool:pytest]` section would be clearer.
**Fix:** Add a minimal `[project]` section or rename to `pytest.ini` / `setup.cfg` depending on intent.

---

_Reviewed: 2026-04-11T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
