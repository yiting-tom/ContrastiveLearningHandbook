# Phase 2: Proxy Tasks Era - Context

**Gathered:** 2026-04-01 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement Instance Discrimination (Wu et al., CVPR 2018) and Invariant Spread (Ye et al., CVPR 2019) as fully working methods that train through `BaseSSLModule`. Deliver `MemoryBank` as a reusable shared infrastructure component. Both methods register in `method_dispatcher` and are documented with DOC-02 docstrings that explain the memory-bank era's core ideas and limitations. Method implementations only — no evaluation tooling, no new shared infrastructure beyond `MemoryBank`.

</domain>

<decisions>
## Implementation Decisions

### Sub-Config Schema Extension
- **D-01:** Add `InstanceDiscriminationConfig` and `InvariantSpreadConfig` Pydantic sub-config classes to `core/config.py`, then add them as `Optional` fields on `TrainConfig` — following the exact same pattern as `SimCLRConfig`, `MoCoConfig`, etc. already present there. The `extra='forbid'` constraint (Phase 1 D-08) means any YAML key not declared as a field raises `ValidationError` at load time; per-method hyper-params (e.g., `n_negatives`, `bank_size`) must be declared here.

### NCE Loss vs. InfoNCELoss Reuse
- **D-02:** The NCE loss for Instance Discrimination (plan 02-02) is a new standalone class in `methods/instance_discrimination/` — it does **not** subclass or wrap `InfoNCELoss`. The Z-normalization semantics (fixed scalar estimated on first batch, ε=1e-7 in denominator) are incompatible with `InfoNCELoss.forward()`, which L2-normalizes inputs internally and has no slot for a fixed Z.
- **D-03:** `InvariantSpreadModule` (plan 02-04) reuses `InfoNCELoss` from `core/losses.py` directly in symmetric mode — no new loss class needed for that method.

### MemoryBank Gradient Handling
- **D-04:** `MemoryBank` is implemented as `nn.Embedding` with `self.weight.requires_grad = False` set immediately after construction. The bank is never sent to the optimizer. Updates happen via direct index assignment (`bank.weight.data[indices] = features`) — not through backprop. This is required because `BaseSSLModule.learnable_params` defaults to `self.parameters()`, which would otherwise include the embedding weight.
- **D-05:** `MemoryBank` is placed in `core/memory_bank.py` (shared infrastructure — used by Instance Discrimination now, and by CMC in a future phase). It is a standalone utility with no method-specific logic.

### Registration Pattern and Module File Layout
- **D-06:** Each method registers itself by calling `register_method()` from within its `__init__.py`: `methods/instance_discrimination/__init__.py` and `methods/invariant_spread/__init__.py`. The top-level `methods/__init__.py` imports both sub-packages to trigger registration before `method_dispatcher` is called.
- **D-07:** Method file layout: `methods/instance_discrimination/module.py` (InstanceDiscriminationModule), `methods/instance_discrimination/losses.py` (NCE loss with fixed Z), `methods/invariant_spread/module.py` (InvariantSpreadModule).

### Claude's Discretion
- Exact NCE sampling strategy for drawing m=4096 negatives from the bank (random without replacement from non-positive indices)
- `MemoryBank` initialization variance (L2-normalized random vectors — uniform on hypersphere)
- Exact config field names and defaults for both sub-configs
- Where to place the weak-augmentation call in `InstanceDiscriminationModule.training_step`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements and method specs
- `.planning/REQUIREMENTS.md` §ERA1-01 — Instance Discrimination full spec (Z-fixing, ε, bank staleness gotcha, bank update timing)
- `.planning/REQUIREMENTS.md` §ERA1-02 — Invariant Spread full spec (batch-size sensitivity, in-batch negatives only, InfoNCELoss reuse)
- `.planning/REQUIREMENTS.md` §INFRA-02 — MemoryBank spec (nn.Embedding, update-by-index, interface)

### Foundation codebase
- `core/base.py` — BaseSSLModule interface (learnable_params, configure_optimizers, on_train_batch_end hook)
- `core/losses.py` — InfoNCELoss (symmetric + asymmetric modes, internal L2-norm — confirms why NCE loss must be separate)
- `core/config.py` — TrainConfig/EvalConfig structure (where to add new Optional sub-config fields)
- `core/dispatcher.py` — register_method() registry pattern
- `core/data.py` — ContrastiveAugmentation strong=False weak path, SSLDataModule interface
- `core/projection.py` — ProjectionHead (reuse for both method encoders)

### Roadmap phase details
- `.planning/ROADMAP.md` §Phase 2 — Goal, success criteria (4 items), pre-specified plan outlines (02-01 through 02-05)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/losses.py` `InfoNCELoss` — reused directly by InvariantSpreadModule in symmetric mode; τ=0.07 matches era-1 defaults
- `core/projection.py` `ProjectionHead` — reuse for both method encoders (2-layer MLP, BN+ReLU intermediate, BN final)
- `core/data.py` `ContrastiveAugmentation(strong=False)` — weak augmentation path already implemented for era-1 methods
- `core/dispatcher.py` `register_method()` — registry pattern ready; phases call this without modifying dispatcher internals
- `core/base.py` `BaseSSLModule` — on_train_batch_end hook for EMA updates; learnable_params override point for excluding bank

### Established Patterns
- Sub-configs are `Optional` fields on `TrainConfig` with `extra='forbid'` (core/config.py pattern)
- Tests use random tensors for unit tests; synthetic ImageFolder for data module tests
- Method packages live in `methods/<method_name>/` with registration in `__init__.py`
- `core/` is for shared infra; `methods/` is for method-specific code

### Integration Points
- `MemoryBank` (new `core/memory_bank.py`) → `InstanceDiscriminationModule.training_step`: bank.get(indices) for negatives, bank.update(indices, features) after loss
- `methods/instance_discrimination/__init__.py` → `core/dispatcher.py`: `register_method("instance_discrimination", InstanceDiscriminationModule)`
- `methods/invariant_spread/__init__.py` → `core/dispatcher.py`: `register_method("invariant_spread", InvariantSpreadModule)`
- `methods/__init__.py` → both sub-packages: explicit imports to trigger registration at startup

</code_context>

<specifics>
## Specific Ideas

- Bank staleness gotcha (keys from earlier encoder snapshots) must be documented in `MemoryBank` class docstring with an explicit cross-reference to why MoCo's queue solves this — this is a ROADMAP success criterion (2).
- Z (normalization constant) must be stored as a `register_buffer` (not a plain Python float) so it survives checkpoint/resume cycles.
- Both methods must be selectable via `method: instance_discrimination` and `method: invariant_spread` in YAML — these are the exact YAML keys.
- `InvariantSpreadModule` docstring must note batch-size sensitivity explicitly (performance degrades below ~256 unlike queue-based methods).

</specifics>

<deferred>
## Deferred Ideas

None — analysis stayed within Phase 2 scope.

</deferred>

---

*Phase: 02-proxy-tasks-era*
*Context gathered: 2026-04-01*
