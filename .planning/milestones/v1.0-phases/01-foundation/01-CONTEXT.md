# Phase 1: Foundation - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the shared infrastructure every SSL method subclass will depend on: `BaseSSLModule`, Pydantic config schema, timm backbone factory, projection head MLP, augmentation pipeline, data module, EMA updater, InfoNCE loss, method dispatcher, and logging wiring. Method implementations begin in Phase 2; this phase only delivers the shared foundation.

</domain>

<decisions>
## Implementation Decisions

### Project Layout
- **D-01:** Top-level `core/` package for all shared foundation code, `methods/` directory where each SSL method gets its own package (e.g., `methods/simclr/`, `methods/moco/`).
- **D-02:** `core/` contains: `base.py` (BaseSSLModule), `backbone.py` (build_backbone), `losses.py` (InfoNCELoss), `projection.py` (ProjectionHead), `data.py` (SSLDataModule + ContrastiveAugmentation), `ema.py` (EMAUpdater), `dispatcher.py` (method_dispatcher), `config.py` (TrainConfig/EvalConfig), `optimizers.py` (LARS).
- **D-03:** Top-level layout: `core/`, `methods/`, `tests/`, `configs/`. No wrapping package (not `ssl_methods/core/`) — `core` and `methods` are importable directly.

### LARS Optimizer
- **D-04:** Implement LARS from scratch in `core/optimizers.py` (~50–60 lines of pure PyTorch). No dependency on `lightly`. Class docstring includes paper reference (https://arxiv.org/abs/1708.03888). This is a tutorial repo — the implementation should be readable.
- **D-05:** `LARS.__init__` signature: `(params, lr, momentum=0.9, weight_decay=1e-6, eta=0.001, exclude_bias_and_norm=True)`.

### Test Fixture Strategy
- **D-06:** Unit tests (backbone, projection head, losses, EMA, config, dispatcher) use random tensors — no I/O, fast, offline-safe.
- **D-07:** `SSLDataModule` and `ContrastiveAugmentation` tests use a temporary synthetic `ImageFolder` (a few dummy `.jpg` images in per-class subdirectories, created by the test fixture and cleaned up after). Tests the actual data-loading path without network access or permanent fixtures.

### Pydantic Config Validation
- **D-08:** `TrainConfig` (and all sub-configs) use `model_config = ConfigDict(extra='forbid')`. Unknown YAML keys raise a `ValidationError` immediately. Typo-catching is important for tutorial users who copy-paste configs.

### Claude's Discretion
- Exact warmup-cosine scheduler implementation (linear warmup then cosine decay is standard)
- LARS `exclude_bias_and_norm` filtering logic (exclude 1-D parameter tensors)
- Synthetic ImageFolder fixture helper (shared pytest fixture or per-test)
- `__init__.py` exports for `core/` (re-export public API vs. leave explicit)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project requirements and constraints
- `.planning/REQUIREMENTS.md` — FOUND-01 through FOUND-10, INFRA-01, INFRA-06: full per-component specifications with gotchas. Every component in this phase has a dedicated requirement entry.
- `.planning/PROJECT.md` — Core value, constraints (Python 3.10+, PyTorch 2.x, Lightning 2.x, minimal deps), key decisions (no Hydra, no OmegaConf).
- `.planning/ROADMAP.md` §Phase 1 — Success criteria (7 items) and pre-specified plan outlines (01-01 through 01-07).

No external ADRs or design docs — all requirements are fully captured in the files above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — project is a blank slate. All foundation components are net-new.

### Established Patterns
- None yet. Phase 1 establishes the patterns that all subsequent phases follow.

### Integration Points
- `core/dispatcher.py` → `methods/*/module.py`: every method registers itself in the dispatcher (Phases 2–8 add entries)
- `core/base.py` → `core/ema.py`: `on_train_batch_end()` calls `EMAUpdater.step()` when a target network is registered
- `core/config.py` → `core/dispatcher.py`: `method_dispatcher(cfg: TrainConfig)` reads `cfg.method` to select the right subclass

</code_context>

<specifics>
## Specific Ideas

- LARS lives in `core/optimizers.py` (not inlined into `base.py`) so it can be imported independently by method implementations that need it.
- The `core/` package should be importable from the project root: `from core.base import BaseSSLModule`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 1 scope.

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-03-30*
