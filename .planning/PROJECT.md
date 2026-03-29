# Contrastive Learning Tutorial Repo

## What This Is

A tutorial repository that implements a wide range of contrastive and self-supervised learning methods (from 2018 to 2021+), built on PyTorch Lightning with timm backbone support. Users configure training via YAML files and can swap models, methods, and datasets through a clean interface — designed for both personal experimentation and educational sharing.

## Core Value

Any contrastive learning method can be implemented by conforming to a shared interface and immediately work with the same dataset pipeline, timm backbone, and Lightning training loop — no boilerplate duplication.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] YAML-based configuration (no Hydra) for selecting method, backbone, and dataset
- [ ] timm backbone integration — any timm model selectable via config
- [ ] PyTorch Lightning training loop shared across all methods
- [ ] Common interface (abstract base class / protocol) that each method implements
- [ ] Custom dataset support (ImageFolder-style or user-defined DataModule)
- [ ] **2018–2019 Proxy Tasks:** Instance Discrimination, Invariant Spread, CPC, CMC, Deep Cluster
- [ ] **2019–2020 MoCo & SimCLR era:** MoCo v1, SimCLR v1, MoCo v2, SimCLR v2, SwAV, CPC v2, InfoMin
- [ ] **2020 No-Negative methods:** BYOL, SimSiam, Barlow Twins
- [ ] **2021 Transformer era:** MoCo v3, DINO, DINO v2, DINO v3
- [ ] **Supervised:** SupCon (Supervised Contrastive Learning)
- [ ] Evaluation suite: linear probing, k-NN, t-SNE, UMAP, fine-tuning, CAM
- [ ] Minimal dependencies — small, auditable tech stack

### Out of Scope

- Hydra / OmegaConf — over-engineered for a tutorial; plain YAML is clearer
- Real-time training dashboards — Lightning's built-in logging is sufficient
- Multi-node distributed training — single-machine GPU focus
- Pre-trained weight downloads / model zoo — users train from scratch or bring their own

## Context

- Tech stack: Python, PyTorch, PyTorch Lightning, timm, YAML (PyYAML or similar)
- Evaluation tools: scikit-learn (linear probe, k-NN), matplotlib/seaborn (t-SNE, UMAP), pytorch-grad-cam (CAM)
- Methods span 2018–2021+; each era represents distinct paradigm shifts worth explaining
- Repo doubles as reference material — code should be readable alongside papers
- No framework magic (Hydra, etc.) — configs are plain dicts loaded from YAML

## Constraints

- **Tech stack**: Minimal — PyTorch Lightning, timm, PyYAML, scikit-learn only; no heavy frameworks
- **Interface**: All methods must implement the shared SSL interface to work with the common training loop
- **Language**: All code, docs, comments in English
- **Compatibility**: Support Python 3.10+, PyTorch 2.x, Lightning 2.x

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch Lightning as training framework | Reduces boilerplate, handles device placement and logging | — Pending |
| timm for backbones | Largest model zoo, clean API, well-maintained | — Pending |
| Plain YAML (no Hydra) | Hydra adds complexity that obscures the tutorial intent | — Pending |
| Shared interface / base class | Lets every method plug into same pipeline — core architectural goal | — Pending |
| Fine granularity phases | Methods span multiple eras; each era/cluster deserves its own phase | — Pending |

---
*Last updated: 2026-03-29 after initial project questioning*
