# Phase 3: SimCLR - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

SimCLR v1 and v2 are working methods that train through `BaseSSLModule` with the correct NT-Xent loss, strong augmentation pipeline, and LARS optimizer — establishing the canonical "two-view in-batch contrastive" pattern that later methods (MoCo v2, BYOL, Barlow Twins) reference. No new shared infrastructure beyond what Phase 1 delivered. Both methods register in `method_dispatcher` and are documented with DOC-02 docstrings.

</domain>

<decisions>
## Implementation Decisions

### NT-Xent Loss

- **D-01:** `SimCLRv1Module` uses `InfoNCELoss` from `core/losses.py` directly with `queue=None` (symmetric mode). No new `NTXentLoss` class — `InfoNCELoss._symmetric_loss` is already NT-Xent semantics. Plan 03-01 delivers unit tests that assert symmetry (`loss(z1, z2) == loss(z2, z1)`) and that identical views yield the minimum possible loss value.

### SimCLRv2 Module Design

- **D-02:** `SimCLRv2Module(SimCLRv1Module)` as a subclass — overrides `build_projector()` to pass `num_layers=3` to `ProjectionHead`. Registered as `simclr_v2` in `method_dispatcher`. Class docstring documents the weight-decay sensitivity difference from v1 (larger projection head requires more regularization). The `num_layers=2 → num_layers=3` switch is controlled by this subclass, not by YAML config.

### Augmentation Validation Script

- **D-03:** Visual inspection script lives at `tools/visualize_augmentations.py`. Standalone CLI script users run as `python tools/visualize_augmentations.py`. Saves a grid of 8 augmented views from one image to confirm strong color jitter (s=1.0) and Gaussian blur are present.

### Sub-Config and Registration (carry-forward)

- **D-04:** `SimCLRConfig` is already defined in `core/config.py` (`temperature=0.5`, `projection_dim=128`). Do not create a new sub-config class. Extend if additional fields are needed (e.g., `num_layers` if wanted as a YAML override on top of the subclass default — but D-02 makes this unnecessary).
- **D-05:** `methods/simclr/__init__.py` calls `register_method("simclr_v1", SimCLRv1Module)` and `register_method("simclr_v2", SimCLRv2Module)`. `methods/__init__.py` imports the `simclr` sub-package to trigger registration — same pattern as Phase 2.

### Claude's Discretion

- Exact `SimCLRv1Module` projector dimensions (REQUIREMENTS.md §ERA2-03 specifies 2048→2048→128; follow that)
- LARS vs AdamW YAML config file names and default values
- Batch-size sensitivity comment wording in YAML configs
- `tools/` directory scaffold (create `tools/__init__.py` or leave as plain scripts dir)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Method specifications (primary)
- `.planning/REQUIREMENTS.md` §ERA2-03 — SimCLR v1 full spec: projection head dims, NT-Xent, strong augmentation gotchas (s=1.0, Gaussian blur), LARS dependency, batch-size sensitivity gotcha
- `.planning/REQUIREMENTS.md` §ERA2-04 — SimCLR v2 full spec: 3-layer head, pretraining-only scope, weight-decay sensitivity gotcha

### Phase roadmap
- `.planning/ROADMAP.md` §Phase 3 — Goal, 5 success criteria, pre-specified plan outlines (03-01 through 03-06)

### Foundation codebase
- `core/losses.py` — `InfoNCELoss` (symmetric mode = NT-Xent; internal L2-norm on inputs; confirms no new loss class needed)
- `core/projection.py` — `ProjectionHead` (num_layers configurable; BN+ReLU on intermediates, BN-only on final layer)
- `core/base.py` — `BaseSSLModule` interface (learnable_params, configure_optimizers, on_train_batch_end hook)
- `core/config.py` — `SimCLRConfig` already defined; `TrainConfig` structure for adding Optional sub-config fields
- `core/data.py` — `ContrastiveAugmentation(strong=True)` with s=1.0 and Gaussian blur already implemented
- `core/optimizers.py` — `LARS` implementation (already exists, no re-implementation needed)
- `core/dispatcher.py` — `register_method()` registry pattern

### Reference method implementation
- `methods/invariant_spread/module.py` — reference for module file layout and registration pattern
- `methods/__init__.py` — shows how top-level import triggers sub-package registration

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `InfoNCELoss(temperature, queue=None)` in `core/losses.py` — call with `queue=None` for symmetric NT-Xent; L2-normalizes inputs internally, no pre-normalization needed in module
- `ProjectionHead(input_dim, hidden_dim, output_dim, num_layers=2)` in `core/projection.py` — pass `num_layers=3` for v2; BN+ReLU on intermediate layers, BN-only on output (correct SSL convention)
- `ContrastiveAugmentation(strong=True)` in `core/data.py` — already uses s=1.0 color jitter and Gaussian blur (strong path); no modifications needed for SimCLR
- `LARS` in `core/optimizers.py` — already implemented from scratch; `BaseSSLModule.configure_optimizers()` dispatches to LARS when `cfg.optimizer == "lars"`
- `register_method()` in `core/dispatcher.py` — same pattern as Phase 2; call from `methods/simclr/__init__.py`

### Established Patterns
- Sub-configs are `Optional` fields on `TrainConfig` with `extra='forbid'` (Phase 1 D-08) — `SimCLRConfig` already registered
- Method packages live in `methods/<method_name>/` with `module.py` + `__init__.py` registration
- `methods/__init__.py` explicit imports trigger sub-package registration at startup
- Tests use random tensors for unit tests; synthetic ImageFolder for integration tests (Phase 1 D-06/D-07)

### Integration Points
- `methods/simclr/module.py` → `core/losses.py`: `InfoNCELoss(cfg.simclr.temperature)`
- `methods/simclr/module.py` → `core/projection.py`: `ProjectionHead(feat_dim, hidden_dim=2048, output_dim=128, num_layers=2)` for v1
- `methods/simclr/__init__.py` → `core/dispatcher.py`: `register_method("simclr_v1", ...)` and `register_method("simclr_v2", ...)`
- `methods/__init__.py` → `methods/simclr`: add `from methods import simclr` import
- New `tools/visualize_augmentations.py` → `core/data.py`: imports `ContrastiveAugmentation` and `SSLDataModule`

</code_context>

<specifics>
## Specific Ideas

- `SimCLRv2Module` docstring must explicitly note: "weight decay scales with projection head depth — v1 fine-tune hyperparameters do not transfer directly to v2" (per ERA2-04 gotcha)
- `tools/visualize_augmentations.py` should save to `tools/output/augmentation_grid.png` (or accept `--output` arg) so users can inspect without a display
- Both YAML configs should include a YAML comment noting batch-size sensitivity (per ROADMAP plan 03-05): `# Note: SimCLR performance degrades sharply below batch_size=256`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 3 scope.

</deferred>

---

*Phase: 03-simclr*
*Context gathered: 2026-04-02*
