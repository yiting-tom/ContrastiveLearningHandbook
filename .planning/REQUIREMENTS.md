# Requirements: Contrastive Learning Tutorial Repo

**Defined:** 2026-05-03
**Milestone:** v1.1 Debt Payoff
**Core Value:** Any contrastive learning method can be implemented by conforming to a shared interface and immediately work with the same dataset pipeline, timm backbone, and Lightning training loop.

## v1.1 Requirements

### train.py Wiring

- [ ] **WIRE-01**: `train.py` correctly wires `InstanceDiscriminationModule` with `IndexedDataset` and `ssl_collate_with_index` so indexed training runs end-to-end
- [ ] **WIRE-02**: `train.py` instantiates `MultiCropDataset` with 2×224 + 6×96 crops for SwAV and DINO methods (not 8 same-size crops)
- [ ] **WIRE-03**: `train.py` routes SupCon stage-2 through `from_stage1_ckpt()` when a stage-1 checkpoint path is provided (not generic `ckpt_path` which trains a random backbone)

### Exports

- [ ] **EXPORT-01**: `PredictorHead`, `SupConLoss`, `MultiCropDataset`, and `method_dispatcher` are importable via `from core import ...` (present in `core/__init__.py __all__`)

### Code Cleanup

- [ ] **CLEAN-01**: Duplicate `InfoMinConfig` definition removed from `config.py` (dead code at lines 72–83)

### Integration Tests

- [ ] **TEST-01**: `@pytest.mark.slow` test asserts BYOL `embedding_std > 0.1` after 5 epochs on real CIFAR-10
- [ ] **TEST-02**: `@pytest.mark.slow` test asserts Barlow Twins `corr_diag_mean > 0.5` after 5 epochs on real CIFAR-10
- [ ] **TEST-03**: `@pytest.mark.slow` test verifies README Quickstart commands run without error (1 epoch, CPU-compatible)

## Future Requirements

No items deferred from v1.1 — this is a closed debt-payoff scope.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New SSL methods (MAE, SimMIM, I-JEPA, etc.) | Not in v1.1 scope — debt-payoff only |
| Experiment tracking (W&B, MLflow) | Deferred to v2+ — Lightning's built-in logging is sufficient for tutorial |
| Multi-node distributed training | Out of scope — single-machine GPU focus |
| Pre-trained weight downloads | Out of scope — users train from scratch |
| SimCLR v2 semi-supervised distillation | Deferred to v2 |
| InfoMin full view-learning | Deferred to v2 |
| DINOv2 training from scratch | Out of scope — hundreds of GPU-days |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| WIRE-01 | — | Pending |
| WIRE-02 | — | Pending |
| WIRE-03 | — | Pending |
| EXPORT-01 | — | Pending |
| CLEAN-01 | — | Pending |
| TEST-01 | — | Pending |
| TEST-02 | — | Pending |
| TEST-03 | — | Pending |

**Coverage:**
- v1.1 requirements: 8 total
- Mapped to phases: 0 (roadmap not yet created)
- Unmapped: 8 ⚠️

---
*Requirements defined: 2026-05-03*
*Last updated: 2026-05-03 after v1.1 milestone start*
