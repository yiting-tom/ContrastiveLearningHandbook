# Phase 7: Transformer Era - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-04-09
**Phase:** 07-transformer-era
**Mode:** discuss
**Areas discussed:** MoCo v3 config, DINO centering vector, DINOv2 loading mechanism, DINOv2 tutorial dataset

## Gray Areas Presented

| Gray Area | Options Offered | Decision |
|-----------|----------------|----------|
| MoCo v3 config class | MoCoV3Config (new) / Extend MoCoConfig | MoCoV3Config (new) |
| DINO centering vector | register_buffer / plain tensor | register_buffer |
| DINOv2 loading mechanism | timm / torch.hub / both-with-flag | timm |
| DINOv2 tutorial dataset | CIFAR-10 / STL-10 / configurable+CIFAR-10 default | Configurable — CIFAR-10 default |

## Discussion Detail

### MoCo v3 Config
- **Presented:** `MoCoV3Config` (new class, no queue_size, momentum=0.99 default) vs. extending `MoCoConfig`
- **User chose:** MoCoV3Config (recommended option)
- **Rationale:** Consistent with project pattern of one config class per method. Avoids `queue_size` bleeding into v3. Clean defaults for temperature=0.2, momentum=0.99, predictor_hidden_dim=4096.

### DINO Centering Vector
- **Presented:** `register_buffer` (checkpointed, matches BN convention) vs. plain tensor (simpler, not checkpointed)
- **User chose:** register_buffer (recommended option)
- **Rationale:** Centering vector is a stateful running statistic that must survive checkpoint/resume. Plain tensor would reset to zeros on resume, causing early-training instability.

### DINOv2 Loading Mechanism
- **Presented:** timm (existing dep) / torch.hub from facebookresearch / both-with-flag
- **User chose:** timm (recommended option)
- **Rationale:** Stays within existing dependency footprint. `timm` supports DINOv2 weights. Tutorial readers don't need to install/configure the facebookresearch/dinov2 repo.

### DINOv2 Tutorial Dataset
- **Presented:** CIFAR-10 fixed / STL-10 fixed / configurable with CIFAR-10 default
- **User chose:** Configurable — CIFAR-10 default
- **Rationale:** Maximally flexible for tutorial readers experimenting with their own data, while CIFAR-10 default keeps zero-setup for readers following the tutorial.

## Corrections Made

No corrections — all recommended options accepted.

## Prior Decisions Applied (No Re-Asking)

- One config class per method: applied → `MoCoV3Config` instead of extending `MoCoConfig`
- `ProjectionHead` and `PredictorHead` reuse over duplication: applied → DINO uses `ProjectionHead` for MLP portion, existing `PredictorHead('standard')` for MoCo v3 prediction head
- `EMAUpdater` shared: applied → DINO teacher EMA uses same `EMAUpdater` as BYOL/MoCo
- `MultiCropDataset` built in Phase 5: applied → DINO consumes it directly, no rebuild needed
