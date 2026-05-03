# Phase 5: SwAV and InfoMin - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-04-08
**Phase:** 05-swav-and-infomin
**Mode:** discuss
**Areas discussed:** MultiCropDataset integration, Crop config location, Prototype normalization hook

---

## Gray Areas Presented

| Area | Options Offered | Chosen |
|------|----------------|--------|
| MultiCropDataset integration | Dataset wrapper injected into SSLDataModule / SSLDataModule detects SwAV config / SwAVModule creates independently | Dataset wrapper injected into SSLDataModule |
| Crop config location | Extend SwAVConfig / New MultiCropConfig field / Top-level TrainConfig fields | Extend SwAVConfig |
| Prototype normalization hook | on_train_batch_end / on_after_backward + manual step / on_before_optimizer_step | on_train_batch_end |

## Decisions Not Discussed (Defaults Applied)

- **InfoMin backbone:** `SimCLRv1Module` — user skipped this area; SimCLR is the natural fit per InfoMin paper's SimCLR-based experiments and simplest reuse path.
- **Sinkhorn-Knopp location:** `methods/swav/losses.py` — follows instance_discrimination pattern (method-specific utilities stay in the method package).

## Key Rationale

- **MultiCropDataset as injected wrapper:** Keeps `SSLDataModule` decoupled from SwAV-specific config. Same `MultiCropDataset` will be reused by DINO in Phase 7 without any changes to `SSLDataModule`.
- **Crop fields in SwAVConfig:** All SwAV hyper-params in one sub-config; DINO gets its own `DINOConfig` with its own crop fields — no cross-method coupling.
- **on_train_batch_end for normalization:** Consistent with established EMA pattern; fires after optimizer step so normalization operates on updated weights. Avoids `automatic_optimization=False` complexity.
