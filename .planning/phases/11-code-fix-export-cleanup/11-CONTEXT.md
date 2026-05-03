# Phase 11: Code Fix & Export Cleanup - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Close 5 specific v1.0 code gaps: 3 `train.py` wiring fixes (InstanceDiscrimination, SwAV/DINO multi-crop, SupCon stage-2), 4 missing `core/__init__.py` exports, and 1 dead-code removal in `config.py`. No new features, no new user-facing capabilities.

</domain>

<decisions>
## Implementation Decisions

### WIRE-01: InstanceDiscrimination wiring (train.py)
- **D-01:** `train.py` wraps the base dataset in `IndexedDataset` before passing to `SSLDataModule` (via the `dataset=` param). Wiring decision stays in train.py where all other method-specific wiring lives.
- **D-02:** `SSLDataModule.train_dataloader()` gets an `isinstance(IndexedDataset)` check that selects `ssl_collate_with_index` — keeps SSLDataModule self-contained, no collate_fn param added to the API.

### WIRE-02: SwAV/DINO multi-crop wiring (train.py)
- **D-03:** Crop sizes for DINO are **hardcoded** in train.py (2×224 + 6×96). No new fields added to `DINOConfig` — avoids config schema churn.
- **D-04:** A **shared helper block** handles both SwAV and DINO: reads `n_large_crops, large_size, n_small_crops, small_size` from `SwAVConfig` when method is `"swav"`, uses hardcoded defaults for `"dino"`. One `if cfg.method in {"swav", "dino"}:` block, not two separate blocks.

### WIRE-03: SupCon stage-2 routing (train.py)
- **D-05:** Detect `cfg.method == "supcon_finetune"`. Call `SupConFinetuneModule.from_stage1_ckpt(args.ckpt_path, cfg)` instead of `method_dispatcher(cfg)`. (Signature: `ckpt_path` is first positional arg, `cfg` is second.)
- **D-06:** If `--ckpt-path` is not provided when `method=supcon_finetune`, **raise a clear error** (`sys.exit(1)` with message: "supcon_finetune requires --ckpt-path pointing to a stage-1 checkpoint"). Prevents silent random-backbone training.
- **D-07:** Do **not** pass `ckpt_path` to `trainer.fit()` for `supcon_finetune`. `from_stage1_ckpt()` loads the backbone; passing it to `trainer.fit()` would incorrectly try to resume Lightning training state from the stage-1 run.

### EXPORT-01: core/__init__.py exports
- **D-08:** Add `PredictorHead` (from `core.projection`), `SupConLoss` (from `core.losses`), `MultiCropDataset` (from `core.data`), and `method_dispatcher` (from `core.dispatcher`) following the existing `try/except ImportError: pass` pattern. Add all 4 to `__all__`.

### CLEAN-01: InfoMinConfig dead code
- **D-09:** Verify that `config.py` has exactly one `InfoMinConfig` definition. If confirmed (current grep shows only line 137), mark CLEAN-01 done with note "no duplicate found — already clean". No code change required.

### Claude's Discretion
- Organization of WIRE-01/02/03 method-specific blocks within `train.py` main() — planner may extract into a helper or keep inline.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — v1.1 requirements: WIRE-01, WIRE-02, WIRE-03, EXPORT-01, CLEAN-01 with exact success criteria
- `.planning/ROADMAP.md` — Phase 11 success criteria (5 specific pass/fail conditions)

### Key source files to read
- `train.py` — entry point being modified; all 3 wiring fixes go here
- `core/data.py` — `IndexedDataset`, `ssl_collate_with_index`, `MultiCropDataset`, `ssl_collate_multi_crop`, `SSLDataModule` (lines 98–360)
- `core/__init__.py` — current exports; add 4 symbols following existing try/except pattern
- `core/config.py` — verify single `InfoMinConfig` definition; `SwAVConfig` crop fields
- `core/projection.py` — `PredictorHead` location for export
- `core/losses.py` — `SupConLoss` location for export
- `core/dispatcher.py` — `method_dispatcher` location for export
- `methods/supcon/module.py` — `SupConFinetuneModule.from_stage1_ckpt()` signature

### Method registry (for train.py method-name checks)
- `methods/instance_discrimination/__init__.py` — registered as `"instance_discrimination"`
- `methods/swav/__init__.py` — registered as `"swav"`
- `methods/dino/__init__.py` — registered as `"dino"`
- `methods/supcon/__init__.py` — `"supcon"` (stage-1) and `"supcon_finetune"` (stage-2)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `IndexedDataset(dataset)` — wraps any dataset to return `(views, label, index)` tuples; in `core/data.py:98`
- `ssl_collate_with_index(batch)` — collate fn for indexed batches; `core/data.py:119`
- `MultiCropDataset` — accepts `dataset, large_crops, small_crops` params; `core/data.py:138`
- `ssl_collate_multi_crop(batch)` — returns list of tensors; `core/data.py:189`
- `SupConFinetuneModule.from_stage1_ckpt(ckpt_path, cfg)` — classmethod; `methods/supcon/module.py:260`

### Established Patterns
- `SSLDataModule` accepts `dataset: Dataset | None` — when provided, skips ImageFolder creation; auto-selects `ssl_collate_multi_crop` when dataset is `MultiCropDataset`
- All method-specific wiring in `train.py` follows the same flat `main()` pattern — no separate dispatch classes
- `core/__init__.py` uses `try/except ImportError: pass` for all imports (parallel-plan safety)
- Method dispatch: `method_dispatcher(cfg)` returns the right module for all methods except `supcon_finetune` stage-2

### Integration Points
- `SSLDataModule.train_dataloader()` — add `isinstance(IndexedDataset)` branch for `ssl_collate_with_index`
- `train.py main()` — add 3 if-blocks before `model = method_dispatcher(cfg)` and `dm = SSLDataModule(...)`:
  1. `supcon_finetune` guard + `from_stage1_ckpt()` path
  2. `instance_discrimination` dataset wrapping
  3. `swav`/`dino` MultiCropDataset construction

</code_context>

<specifics>
## Specific Ideas

- The `SwAVConfig` crop defaults (`n_large_crops=2, large_size=224, n_small_crops=6, small_size=96`) match DINO paper defaults — reuse them as hardcoded constants for DINO in train.py.
- Error message for missing ckpt_path: `"supcon_finetune requires --ckpt-path pointing to a stage-1 checkpoint"` (suggested exact text).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 11-Code Fix & Export Cleanup*
*Context gathered: 2026-05-03*
