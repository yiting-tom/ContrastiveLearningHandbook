# Contrastive Learning Tutorial Repo

## What This Is

A tutorial repository implementing the full arc of contrastive and self-supervised learning methods (2018–2021+), built on PyTorch Lightning with timm backbone support. All 14 v1 methods are implemented, documented with paper-accurate gotchas, and connected via a shared `BaseSSLModule` interface, YAML-driven config system, and a complete evaluation suite. Designed for both personal experimentation and educational publication.

## Core Value

Any contrastive learning method can be implemented by conforming to a shared interface and immediately work with the same dataset pipeline, timm backbone, and Lightning training loop — no boilerplate duplication.

## Requirements

### Validated (v1.0)

- ✓ YAML-based configuration (no Hydra) for selecting method, backbone, and dataset — v1.0
- ✓ timm backbone integration — any timm model selectable via config — v1.0
- ✓ PyTorch Lightning training loop shared across all methods — v1.0
- ✓ Common interface (abstract base class) that each method implements — v1.0
- ✓ Custom dataset support (ImageFolder-style DataModule) — v1.0
- ✓ **2018–2019 Proxy Tasks:** Instance Discrimination, Invariant Spread — v1.0
- ✓ **2019–2020 MoCo & SimCLR era:** MoCo v1/v2, SimCLR v1/v2, SwAV, InfoMin — v1.0
- ✓ **2020 No-Negative methods:** BYOL, SimSiam, Barlow Twins — v1.0
- ✓ **2021 Transformer era:** MoCo v3, DINO, DINOv2 (feature extraction) — v1.0
- ✓ **Supervised:** SupCon (Supervised Contrastive Learning) — v1.0
- ✓ Evaluation suite: linear probing, k-NN, t-SNE, UMAP, fine-tuning, CAM — v1.0
- ✓ Minimal dependencies — small, auditable tech stack — v1.0

### Active (v1.1 targets)

- [ ] Fix `train.py` → `InstanceDiscriminationModule` wiring (IndexedDataset + ssl_collate_with_index)
- [ ] Fix `train.py` → SwAV/DINO multi-crop wiring (instantiate MultiCropDataset with 2×224 + 6×96)
- [ ] Fix `train.py` → SupCon stage-2 routing (call `from_stage1_ckpt()` instead of ckpt_path)
- [ ] Add `PredictorHead`, `SupConLoss`, `MultiCropDataset`, `method_dispatcher` to `core/__init__.py __all__`
- [ ] Human verification: BYOL embedding_std >0.1 on real CIFAR-10 for 5 epochs
- [ ] Human verification: Barlow Twins corr_diag_mean >0.5 on CIFAR-10 by epoch 5
- [ ] Human verification: README Quickstart end-to-end GPU run

### Out of Scope

- Hydra / OmegaConf — over-engineered for a tutorial; plain YAML is clearer
- Real-time training dashboards — Lightning's built-in logging is sufficient
- Multi-node distributed training — single-machine GPU focus
- Pre-trained weight downloads / model zoo — users train from scratch or bring their own
- SimCLR v2 semi-supervised distillation stage — deferred to v2 requirements
- InfoMin full view-learning (semi-supervised) — deferred to v2 requirements
- DINOv2 training from scratch — hundreds of GPU-days; not tutorial-viable

## Context

- **v1.0 shipped 2026-05-03** — 34 days, 274 commits, ~112,700 LOC Python
- Tech stack: Python 3.10+, PyTorch 2.x, PyTorch Lightning 2.x, timm 1.x, PyYAML, scikit-learn, pytorch-grad-cam
- Evaluation tools: FAISS (k-NN), scikit-learn (linear probe, t-SNE), umap-learn (UMAP), pytorch-grad-cam (CAM)
- Methods span 2018–2021+; each era represents distinct paradigm shifts documented in `docs/tutorial.md`
- All 14 v1 methods implemented and unit-tested; 13/13 e2e pipeline tests GREEN
- Known integration gaps in `train.py` for 3 niche methods (InstanceDiscrimination, SwAV/DINO multi-crop, SupCon stage-2) — deferred to v1.1

## Constraints

- **Tech stack**: Minimal — PyTorch Lightning, timm, PyYAML, scikit-learn only; no heavy frameworks
- **Interface**: All methods must implement the shared SSL interface to work with the common training loop
- **Language**: All code, docs, comments in English
- **Compatibility**: Python 3.10+, PyTorch 2.x, Lightning 2.x

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch Lightning as training framework | Reduces boilerplate, handles device placement and logging | ✓ Good — all 14 methods share identical training loop |
| timm for backbones | Largest model zoo, clean API, well-maintained | ✓ Good — `build_backbone()` handles ResNets and ViTs identically |
| Plain YAML (no Hydra) | Hydra adds complexity that obscures the tutorial intent | ✓ Good — `extra='forbid'` on Pydantic catches typos immediately |
| Shared interface / base class | Lets every method plug into same pipeline — core architectural goal | ✓ Good — dispatcher + registry enables zero-boilerplate method addition |
| Fine granularity phases (10+) | Methods span multiple eras; each era/cluster deserves its own phase | ✓ Good — parallel execution within phases was efficient |
| `nn.Embedding` for MemoryBank | Indexed lookup, `requires_grad=False`, L2-normalized storage | ✓ Good — clean indexed update API |
| `NCELossWithFixedZ` standalone | Incompatible Z-normalization semantics with InfoNCELoss | ✓ Good — no subclassing confusion |
| Queue stored as `[dim, queue_size]` | Direct matrix multiply with query vectors | ✓ Good — no transpose needed in loss computation |
| SimCLRv2 inherits SimCLRv1 | Override only `build_projector()` for 3-layer head | ✓ Good — minimal variant pattern |
| `SupConLoss(labels=None)` degenerates to SimCLR | Clean design for gradual supervision | ✓ Good — unit-tested equivalence |
| EigenCAM as SSL default | No classifier needed; first PC of final layer output | ✓ Good — works on all methods without modification |
| LARS from scratch (~60 lines) | No lightly/torchlars dependency; tutorial-readable | ✓ Good — reduces dependency surface |
| Phase 10.1 inserted post-audit | Closed critical eval-script integration blockers | ✓ Good — 13/13 e2e tests GREEN |

---
*Last updated: 2026-05-03 after v1.0 milestone — 11 phases, 57 plans, 34/40 requirements fully satisfied. Next: /gsd-new-milestone for v1.1.*
