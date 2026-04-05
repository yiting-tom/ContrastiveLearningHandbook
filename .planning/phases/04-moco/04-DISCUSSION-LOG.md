# Phase 4: MoCo - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-04-05
**Phase:** 04-moco
**Mode:** discuss
**Areas discussed:** v1 projection head fidelity, MomentumQueue file location, EMAUpdater re-use for fixed momentum, Shuffled BN handling depth

---

## Gray Areas Presented

| Area | Options Offered | Notes |
|------|----------------|-------|
| v1 projection head | `nn.Linear` (paper-faithful) / `ProjectionHead(num_layers=1)` (API-consistent) | User asked for clarification before answering |
| MomentumQueue location | `core/queue.py` (new file) / `core/memory_bank.py` (extend existing) | — |
| EMA pattern | `EMAUpdater(base=end=0.999)` (reuse hook) / inline fixed EMA in module | — |
| Shuffled BN depth | Docstring-only note / placeholder stub with NotImplementedError | — |

---

## Decisions Made

### v1 Projection Head
- **Question:** For MoCo v1, which projection head design?
- **User clarification:** "follow the paper"
- **Decision:** `nn.Linear(feat_dim, 128)` — bare linear projection, no BN. Paper-faithful.
- **Rationale:** Makes the v1→v2 upgrade (add 2-layer MLP) clearly visible in the code diff.

### MomentumQueue File Location
- **Question:** Where should MomentumQueue live?
- **Answer:** `core/queue.py` (Recommended)
- **Rationale:** MemoryBank (update-by-index) and MomentumQueue (FIFO) are conceptually different; separate files improves discoverability.

### EMA Momentum Pattern
- **Question:** How should MoCo's fixed EMA momentum be implemented?
- **Answer:** `EMAUpdater(base=0.999, end=0.999, ...)` (Recommended)
- **Rationale:** Consistent with BYOL/DINO hook pattern; setting base=end gives flat schedule.

### Shuffled BN Handling
- **Question:** How deep should the shuffled BN handling go?
- **Answer:** Docstring-only note (Recommended)
- **Rationale:** Tutorial targets single-GPU; no dead code stubs needed.

---

## No Corrections

All recommended options were accepted. No scope creep encountered.
