# Phase 5: SwAV and InfoMin - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

SwAV's online clustering with multi-crop is working; `MultiCropDataset` is a reusable component (also consumed by DINO in Phase 7); InfoMin is presented as an augmentation-policy demonstration on top of the existing SimCLR backbone. Method implementations only — no new shared infrastructure beyond `MultiCropDataset` and the `SwAVConfig` crop fields.

</domain>

<decisions>
## Implementation Decisions

### MultiCropDataset Integration

- **D-01:** `MultiCropDataset` is a `torch.utils.data.Dataset` **wrapper** — it wraps any existing dataset and applies `n_large_crops` large augmentations and `n_small_crops` small augmentations per sample, yielding a list of `n_large + n_small` tensors. It lives in `core/data.py` alongside `SSLDataModule`.
- **D-02:** `SSLDataModule` accepts a pre-built dataset via its `dataset` parameter (or detects a `MultiCropDataset` instance). When multi-crop is active, `SSLDataModule` uses a multi-crop-aware collate function instead of `ssl_collate_fn`. `SSLDataModule` does NOT read `SwAVConfig` directly — the caller (or `SwAVModule.setup`) is responsible for constructing `MultiCropDataset` and passing it in.
- **D-03:** `MultiCropDataset.__init__` signature: `(dataset, n_large_crops, large_size, n_small_crops, small_size, strong=True)`. It instantiates two `ContrastiveAugmentation` instances internally (one at `large_size`, one at `small_size`).

### SwAVConfig Crop Fields

- **D-04:** Extend `SwAVConfig` (in `core/config.py`) with crop parameters: `n_large_crops: int = 2`, `large_size: int = 224`, `n_small_crops: int = 6`, `small_size: int = 96`, `temperature: float = 0.1`, `epsilon: float = 0.05`. All SwAV hyper-params in one sub-config; YAML keys are `swav.n_large_crops`, `swav.large_size`, etc.
- **D-05:** The `extra='forbid'` constraint already on `_StrictBase` applies to the updated `SwAVConfig` — no additional validation needed.

### Prototype Normalization

- **D-06:** Prototype L2-renormalization happens in `on_train_batch_end` — the same hook used by `EMAUpdater` in MoCo/BYOL. This fires after `optimizer.step()` completes, so the renormalization operates on updated prototype weights. Do NOT use `on_after_backward` (fires before optimizer step). `automatic_optimization=True` is preserved — no manual optimizer management needed.
- **D-07:** Prototype gradient freezing during `freeze_prototypes_epochs` happens in `on_before_optimizer_step`: zero the prototype weight gradient (`self.prototype_layer.weight.grad.zero_()`) if the current epoch < `freeze_prototypes_epochs`. This separates freeze logic (pre-step) from normalization logic (post-step).

### Sinkhorn-Knopp Location

- **D-08:** `sinkhorn_knopp(scores, n_iters=3, epsilon=0.05)` is a **standalone function in `methods/swav/losses.py`** — same pattern as the NCE loss in `methods/instance_discrimination/losses.py`. It is SwAV-specific and doesn't belong in `core/losses.py`.

### SwAVModule Structure

- **D-09:** `SwAVModule(BaseSSLModule)` lives in `methods/swav/module.py`. The prototype layer is `nn.Linear(feat_dim, n_prototypes, bias=False)` — this is the prototype weight matrix C in the paper. `learnable_params` property is overridden to include the prototype layer parameters alongside backbone + projector.
- **D-10:** `methods/swav/__init__.py` registers `register_method("swav", SwAVModule)`. `methods/__init__.py` imports `methods.swav` to trigger registration — same pattern as MoCo, SimCLR.

### InfoMin Module Design

- **D-11:** `InfoMinModule(SimCLRv1Module)` subclasses `SimCLRv1Module` — overrides `build_augmentation()` to return a `ContrastiveAugmentation` instance with aggressive color jitter (s=1.5), random grayscale (p=0.4), and **no** Gaussian blur. The backbone, projection head, and NT-Xent loss are inherited unchanged.
- **D-12:** `InfoMinModule` registers as `"infomin"` in `method_dispatcher`. YAML config selects it via `method: infomin`.
- **D-13:** The comparison script lives at `tools/compare_augmentations.py` (standalone CLI). It saves a side-by-side grid: left panel shows standard SimCLR augmentation, right panel shows InfoMin augmentation, for the same source image. Output: `tools/output/augmentation_comparison.png`.

### Claude's Discretion

- Exact augmentation parameters for InfoMin (jitter s=1.5, grayscale p=0.4 are starting points — fine-tune within the "minimal MI" spirit)
- YAML comment wording for the 8-crop memory usage warning
- `ssl_collate_multi_crop` implementation details (list of per-crop tensors vs. dict)
- Whether `MultiCropDataset` emits labels alongside the crop list (keep them for eval compatibility)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Method specifications
- `.planning/REQUIREMENTS.md` §ERA2-05 — SwAV full spec: Sinkhorn-Knopp, prototype constraints, multi-crop strategy, doubly-stochastic code matrix, 3-gotcha list
- `.planning/REQUIREMENTS.md` §ERA2-06 — InfoMin full spec: augmentation-policy interpretation, minimal-MI principle, full view-learning deferred to v2
- `.planning/REQUIREMENTS.md` §INFRA-04 — MultiCropDataset spec: interface, n_large_crops/n_small_crops, shared by SwAV and DINO

### Phase roadmap
- `.planning/ROADMAP.md` §Phase 5 — Goal, 5 success criteria, pre-specified plan outlines (05-01 through 05-07)

### Foundation codebase
- `core/config.py` — `SwAVConfig` (extend with crop fields per D-04); `TrainConfig.swav` field already present
- `core/data.py` — `ContrastiveAugmentation`, `MultiViewTransform`, `SSLDataModule`, `ssl_collate_fn` (reference for multi-crop collate design)
- `core/base.py` — `BaseSSLModule`: `on_train_batch_end` hook (already used for EMA); `on_before_optimizer_step` hook; `learnable_params` override point
- `core/projection.py` — `ProjectionHead` (reuse for SwAV projector: feat_dim → 128, 2-layer MLP)
- `core/losses.py` — `InfoNCELoss` (reused by `InfoMinModule` via SimCLRv1Module inheritance)
- `core/dispatcher.py` — `register_method()` registry pattern
- `methods/simclr/module.py` — `SimCLRv1Module` (base class for InfoMin; `build_augmentation()` override point)
- `methods/instance_discrimination/losses.py` — reference for method-specific loss/utility function placement (pattern for `methods/swav/losses.py`)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ContrastiveAugmentation(size, strong)` in `core/data.py` — instantiate two: one at `large_size=224`, one at `small_size=96`. The `size` parameter already controls crop size.
- `ProjectionHead(input_dim, hidden_dim, output_dim, num_layers=2)` in `core/projection.py` — use for SwAV projector (standard 2-layer, 128-dim output matching original paper)
- `InfoNCELoss` in `core/losses.py` — inherited by `InfoMinModule` via `SimCLRv1Module`; no changes needed
- `on_train_batch_end` in `core/base.py` — already called for EMA; add prototype renormalization here in `SwAVModule`
- `on_before_optimizer_step` — available in `BaseSSLModule` (Lightning hook); use to zero prototype grads during freeze epochs

### Established Patterns
- Method sub-configs are `Optional` fields on `TrainConfig` with `extra='forbid'` — `SwAVConfig` already registered at `TrainConfig.swav`
- Method packages: `methods/swav/module.py` + `methods/swav/losses.py` + `methods/swav/__init__.py`
- `learnable_params` property must exclude momentum encoder params (from MoCo/BYOL pattern); for SwAV it must *include* prototype params
- Tests: random-tensor unit tests for loss functions; `SSLDataModule` tests use synthetic ImageFolder

### Integration Points
- `MultiCropDataset` (new `core/data.py`) → `SSLDataModule`: passed as a pre-built dataset; SSLDataModule uses a multi-crop collate function when it detects a `MultiCropDataset`
- `SwAVModule.setup()` → `MultiCropDataset`: constructs the wrapper from `cfg.swav` crop fields and passes to `SSLDataModule`
- `methods/swav/module.py` → `methods/swav/losses.py`: imports `sinkhorn_knopp`
- `methods/swav/__init__.py` → `core/dispatcher.py`: `register_method("swav", SwAVModule)`
- `methods/infomin/__init__.py` → `core/dispatcher.py`: `register_method("infomin", InfoMinModule)`
- `methods/__init__.py` → both sub-packages: add `from methods import swav` and `from methods import infomin`

</code_context>

<specifics>
## Specific Ideas

- SwAV YAML config must include a comment: `# Memory usage with 8 crops is ~4× SimCLR — reduce batch_size accordingly`
- Sinkhorn-Knopp test must assert: row sums ≈ 1/K and column sums ≈ 1/B (doubly stochastic) per ROADMAP success criterion 2
- `SwAVModule` docstring gotcha list must include: (1) prototypes must be frozen for first `freeze_prototypes_epochs` epochs; (2) prototype vectors must be L2-normalized after every optimizer step; (3) Sinkhorn-Knopp code matrix must be doubly stochastic — use `epsilon=0.05`; (4) batch size reduction required for 8-crop configuration
- InfoMin docstring must explain the minimal-MI principle: "views should share task-relevant information but minimize mutual information beyond that — more aggressive augmentation removes spurious correlations"
- `tools/compare_augmentations.py` should accept `--image` path and `--output` path arguments

</specifics>

<deferred>
## Deferred Ideas

- Full InfoMin view-learning framework (semi-supervised, requires labeled subset) — V2-06 in REQUIREMENTS.md
- Adaptive crop sizing / focal crops — V2-07 in REQUIREMENTS.md

</deferred>

---

*Phase: 05-swav-and-infomin*
*Context gathered: 2026-04-08*
