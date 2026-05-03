# Phase 4: MoCo - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

MoCo v1 and v2 are working methods that train through `BaseSSLModule` with the correct momentum encoder, FIFO queue, and documented shuffled-BN limitation — establishing the queue-based contrastive pattern and its evolution from v1 to v2. `MomentumQueue` is a standalone reusable component in `core/`. Both methods register in `method_dispatcher` and are documented with DOC-02 docstrings.

</domain>

<decisions>
## Implementation Decisions

### MoCo v1 Projection Head

- **D-01:** `MoCoV1Module` uses `nn.Linear(feat_dim, 128)` — a bare linear projection with no BN, matching the original MoCo v1 paper exactly. This makes the v1→v2 upgrade (replace linear with 2-layer MLP) clearly visible in the code diff. Do not use `ProjectionHead(num_layers=1)` for v1 — that would add BN not present in the paper and blur the architectural distinction.

### MoCo v2 Projection Head

- **D-02:** `MoCoV2Module(MoCoV1Module)` subclasses v1 and overrides `build_projector()` to return `ProjectionHead(feat_dim, 2048, 128, num_layers=2)`. This is the "add MLP head" upgrade from the MoCo v2 tech report. The docstring documents this as a 5-line diff from v1.

### MomentumQueue File Location

- **D-03:** `MomentumQueue` lives in `core/queue.py` — a new file separate from `core/memory_bank.py`. MemoryBank (update-by-index, used by Instance Discrimination) and MomentumQueue (FIFO, used by MoCo v1/v2) are conceptually different data structures. Separate files improve discoverability for tutorial readers.

### EMA Momentum Pattern

- **D-04:** Use `EMAUpdater(base_momentum=0.999, end_momentum=0.999, total_steps=...)` to get a fixed-momentum schedule. Setting `base == end` gives a flat schedule (no cosine ramp) while still routing through the shared `on_train_batch_end` hook — consistent with BYOL and DINO. Do not inline a manual EMA update; the base class hook pattern is the established convention.

### Shuffled BN Handling

- **D-05:** Document the shuffled-BN requirement in the `MoCoV1Module` class docstring only. No code placeholder or stub. The tutorial targets single-GPU; multi-GPU shuffled BN is documented as a limitation in the gotcha list. The docstring must reference the original paper's shuffled-BN technique and explain that the BN-leakage problem is what MoCo's shuffled BN solves.

### Module and Package Structure

- **D-06:** Both `MoCoV1Module` and `MoCoV2Module` live in `methods/moco/module.py` — same layout as SimCLR (one package, one module file for the method family). `methods/moco/__init__.py` registers both as `moco_v1` and `moco_v2` via `register_method()`. `methods/__init__.py` imports the `moco` sub-package to trigger registration.

### Loss Function

- **D-07:** Use `InfoNCELoss._asymmetric_loss` (queue mode) directly — call `InfoNCELoss(temperature=moco_cfg.temperature)` and pass the queue via the `queue` argument. No new loss class. Keys are detached before the loss; queue is updated after the loss computation (not before).

### Claude's Discretion

- `MomentumQueue` initialization: `torch.randn` (L2-normalized) as specified in plan 04-01
- Queue pointer wrap-around implementation details
- `assert_no_ema_in_optimizer` utility placement (test helper in `tests/` or inline assertion in module)
- YAML config defaults (AdamW vs SGD, default LR, scheduler choice for v2 cosine)
- Exact wording of m=0.9 vs m=0.999 sensitivity gotcha in docstring

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Method specifications (primary)
- `.planning/REQUIREMENTS.md` §ERA2-01 — MoCo v1 full spec: FC projection, queue K=65536, momentum m=0.999, shuffled-BN requirement, MomentumQueue interface, m=0.9 sensitivity gotcha
- `.planning/REQUIREMENTS.md` §ERA2-02 — MoCo v2 full spec: 2-layer MLP head, Gaussian blur augmentation, cosine LR schedule, documented as "5-line diff from v1"
- `.planning/REQUIREMENTS.md` §INFRA-03 — MomentumQueue spec: FIFO buffer, `get_negatives()` and `update(keys)` interface (note: plan 04-01 uses `enqueue_dequeue(keys)` — plan takes precedence as implementation detail)

### Phase roadmap
- `.planning/ROADMAP.md` §Phase 4 — Goal, 5 success criteria, pre-specified plan outlines (04-01 through 04-06)

### Foundation codebase
- `core/ema.py` — `EMAUpdater` with cosine-scheduled momentum; `base_momentum=end_momentum=0.999` gives flat schedule for MoCo
- `core/losses.py` — `InfoNCELoss` with `_asymmetric_loss` (queue mode already implemented); check interface: `forward(z_i, z_j, queue)`
- `core/projection.py` — `ProjectionHead` (for v2 MLP head); v1 uses `nn.Linear` instead
- `core/base.py` — `BaseSSLModule` EMA hook: `self.ema_updater`, `self._online_params`, `self._target_params` must be set by subclass
- `core/config.py` — `MoCoConfig` already defined: `temperature=0.07`, `queue_size=65536`, `momentum=0.999`; `TrainConfig.moco` field already present
- `core/memory_bank.py` — reference for `nn.Embedding`-based buffer pattern (MomentumQueue will be in separate `core/queue.py`)
- `core/dispatcher.py` — `register_method()` registry pattern

### Reference method implementation
- `methods/simclr/module.py` — reference for module layout, subclass pattern (v2 subclasses v1), and `build_projector()` override convention
- `methods/__init__.py` — shows how top-level import triggers sub-package registration

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `InfoNCELoss(temperature, queue=None)` in `core/losses.py` — queue mode (`_asymmetric_loss`) already implemented and tested; pass `queue=self.momentum_queue.get_negatives()` and `z_j` as positive key
- `EMAUpdater(base_momentum, end_momentum, total_steps)` in `core/ema.py` — set `base=end=0.999` for flat schedule; assign to `self.ema_updater` and set `self._online_params` / `self._target_params` in `__init__`
- `ProjectionHead(input_dim, hidden_dim, output_dim, num_layers)` in `core/projection.py` — used by MoCo v2 only (v1 uses bare `nn.Linear`)
- `register_method()` in `core/dispatcher.py` — same registration pattern as SimCLR / Instance Discrimination

### Established Patterns
- Method subclass relationship: `MoCoV2Module(MoCoV1Module)` — override `build_projector()` only (same as `SimCLRv2Module(SimCLRv1Module)`)
- EMA setup in `__init__`: `copy.deepcopy(backbone)`, then `backbone_ema.requires_grad_(False)`, then assign `self.ema_updater`, `self._online_params`, `self._target_params`
- `learnable_params` property must be overridden to exclude EMA target params (see `BaseSSLModule` docstring: "MUST have `requires_grad=False`")
- Method sub-configs accessed via `cfg.moco or MoCoConfig()` fallback pattern (same as `cfg.simclr or SimCLRConfig()`)

### Integration Points
- `methods/moco/module.py` → `core/queue.py`: `MomentumQueue(queue_size, dim)`
- `methods/moco/module.py` → `core/losses.py`: `InfoNCELoss(cfg.moco.temperature)` with queue argument
- `methods/moco/module.py` → `core/ema.py`: `EMAUpdater(0.999, 0.999, total_steps)`
- `methods/moco/__init__.py` → `core/dispatcher.py`: `register_method("moco_v1", MoCoV1Module)` and `register_method("moco_v2", MoCoV2Module)`
- `methods/__init__.py` → `methods/moco`: add `from methods import moco` import (same pattern as SimCLR, Instance Discrimination)
- New `core/queue.py` → no existing dependencies; exposes `MomentumQueue` class

</code_context>

<specifics>
## Specific Ideas

- MoCo v1 uses `nn.Linear(feat_dim, 128)` — not `ProjectionHead`. The docstring should explicitly note this follows the paper exactly and that the move to MLP in v2 is a deliberate architectural upgrade.
- MoCo v2 docstring must say "this is a 5-line diff from v1" (per ROADMAP plan 04-05) — the subclass makes this literal.
- `MoCoV1Module` gotcha list must include: (1) shuffled BN for multi-GPU — not implemented, single-GPU only; (2) m=0.9 produces significantly worse results than m=0.999; (3) queue must be updated after loss computation, not before.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 4 scope.

</deferred>

---

*Phase: 04-moco*
*Context gathered: 2026-04-05*
