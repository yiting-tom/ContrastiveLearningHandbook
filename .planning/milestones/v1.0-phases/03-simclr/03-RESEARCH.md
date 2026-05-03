# Phase 3: SimCLR - Research

**Researched:** 2026-04-05
**Domain:** SimCLR v1/v2 contrastive learning, NT-Xent loss, LARS optimizer integration
**Confidence:** HIGH

## Summary

Phase 3 implements SimCLR v1 and v2 as the canonical "two-view in-batch contrastive" pattern. The foundation infrastructure from Phase 1 provides nearly everything needed: `InfoNCELoss` in symmetric mode already implements NT-Xent semantics, `ProjectionHead` supports configurable depth (num_layers=2 for v1, num_layers=3 for v2), `ContrastiveAugmentation(strong=True)` already uses s=1.0 color jitter with Gaussian blur, `LARS` optimizer is already implemented and wired into `BaseSSLModule.configure_optimizers()`, and `SimCLRConfig` is already defined in `core/config.py`.

The primary implementation work is: (1) writing the `SimCLRv1Module` and `SimCLRv2Module` classes following the established `InvariantSpreadModule` pattern, (2) creating YAML configs with LARS and AdamW variants, (3) writing unit tests for NT-Xent symmetry and the v2 projection head depth change, (4) creating a visualization script at `tools/visualize_augmentations.py`, and (5) adding DOC-02 docstrings. No new core infrastructure is needed.

**Primary recommendation:** Follow the InvariantSpreadModule pattern exactly. SimCLRv1Module uses `InfoNCELoss(queue=None)` for NT-Xent, SimCLRv2Module subclasses v1 and overrides `build_projector()` to pass `num_layers=3`. LARS is already wired -- just set `optimizer: lars` in YAML.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `SimCLRv1Module` uses `InfoNCELoss` from `core/losses.py` directly with `queue=None` (symmetric mode). No new `NTXentLoss` class -- `InfoNCELoss._symmetric_loss` is already NT-Xent semantics. Plan 03-01 delivers unit tests that assert symmetry (`loss(z1, z2) == loss(z2, z1)`) and that identical views yield the minimum possible loss value.
- **D-02:** `SimCLRv2Module(SimCLRv1Module)` as a subclass -- overrides `build_projector()` to pass `num_layers=3` to `ProjectionHead`. Registered as `simclr_v2` in `method_dispatcher`. Class docstring documents the weight-decay sensitivity difference from v1 (larger projection head requires more regularization). The `num_layers=2 -> num_layers=3` switch is controlled by this subclass, not by YAML config.
- **D-03:** Visual inspection script lives at `tools/visualize_augmentations.py`. Standalone CLI script users run as `python tools/visualize_augmentations.py`. Saves a grid of 8 augmented views from one image to confirm strong color jitter (s=1.0) and Gaussian blur are present.
- **D-04:** `SimCLRConfig` is already defined in `core/config.py` (`temperature=0.5`, `projection_dim=128`). Do not create a new sub-config class.
- **D-05:** `methods/simclr/__init__.py` calls `register_method("simclr_v1", SimCLRv1Module)` and `register_method("simclr_v2", SimCLRv2Module)`. `methods/__init__.py` imports the `simclr` sub-package to trigger registration -- same pattern as Phase 2.

### Claude's Discretion
- Exact `SimCLRv1Module` projector dimensions (REQUIREMENTS.md ERA2-03 specifies 2048->2048->128; follow that)
- LARS vs AdamW YAML config file names and default values
- Batch-size sensitivity comment wording in YAML configs
- `tools/` directory scaffold (create `tools/__init__.py` or leave as plain scripts dir)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within Phase 3 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ERA2-03 | SimCLR v1: In-batch NT-Xent loss (symmetric), shared encoder, 2-layer MLP projection head (128-dim output), loss on `z`, eval on `h`, LARS optimizer, strong augmentation (s=1.0, Gaussian blur), batch-size sensitivity | `InfoNCELoss(queue=None)` already implements symmetric NT-Xent; `ProjectionHead(num_layers=2)` exists; `ContrastiveAugmentation(strong=True)` has s=1.0; `LARS` exists in `core/optimizers.py`; `BaseSSLModule.configure_optimizers()` dispatches to LARS when `optimizer: lars` |
| ERA2-04 | SimCLR v2: 3-layer MLP projection head (depth 3 instead of 2), pretraining stage only, weight-decay sensitivity gotcha | `ProjectionHead(num_layers=3)` is supported; `SimCLRv2Module` subclasses v1 and overrides `build_projector()` per D-02 |
</phase_requirements>

## Standard Stack

### Core (all already available in project)
| Library | Purpose | Status | Source |
|---------|---------|--------|--------|
| `core/losses.py::InfoNCELoss` | NT-Xent loss via symmetric mode (`queue=None`) | EXISTS | Verified in codebase |
| `core/projection.py::ProjectionHead` | 2-layer (v1) and 3-layer (v2) MLP | EXISTS | Verified: supports `num_layers` param |
| `core/data.py::ContrastiveAugmentation` | Strong augmentation with s=1.0 color jitter + Gaussian blur | EXISTS | Verified: `strong=True` path |
| `core/optimizers.py::LARS` | Layer-wise Adaptive Rate Scaling for large batches | EXISTS | Verified in codebase |
| `core/config.py::SimCLRConfig` | `temperature=0.5`, `projection_dim=128` | EXISTS | Verified in codebase |
| `core/base.py::BaseSSLModule` | Abstract base with optimizer dispatch, LR scheduling, EMA hook | EXISTS | Verified in codebase |
| `core/backbone.py::build_backbone` | timm factory returning `(backbone, feat_dim)` | EXISTS | Verified in codebase |
| `core/dispatcher.py::register_method` | Method registry pattern | EXISTS | Verified in codebase |

### Supporting (for visualization script)
| Library | Purpose | When to Use |
|---------|---------|-------------|
| `matplotlib` | Save augmentation grid in `tools/visualize_augmentations.py` | Plan 03-04 |
| `torchvision` | Load sample image for augmentation visualization | Plan 03-04 |

**No new packages need to be installed.** All dependencies are already in the project.

## Architecture Patterns

### Project Structure for Phase 3
```
methods/
  simclr/
    __init__.py       # register_method("simclr_v1", ...) + register_method("simclr_v2", ...)
    module.py         # SimCLRv1Module + SimCLRv2Module
tools/
  visualize_augmentations.py   # Standalone CLI script (D-03)
configs/
  simclr_v1_resnet18.yaml     # AdamW default
  simclr_v1_resnet50_lars.yaml # LARS variant
  simclr_v2_resnet18.yaml     # v2 with 3-layer head
tests/
  test_simclr.py              # NT-Xent symmetry, training, dispatcher, configs
```

### Pattern 1: Method Module (follow InvariantSpreadModule exactly)
**What:** Each method subclasses `BaseSSLModule`, implements `build_projector()` and `training_step()`.
**When to use:** Every method module.
**Example:**
```python
# Source: methods/invariant_spread/module.py (verified in codebase)
class SimCLRv1Module(BaseSSLModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        self.backbone, self.feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
        self.projector = self.build_projector()
        simclr_cfg = cfg.simclr or SimCLRConfig()
        self.loss_fn = InfoNCELoss(temperature=simclr_cfg.temperature)

    def build_projector(self) -> nn.Module:
        simclr_cfg = self.cfg.simclr or SimCLRConfig()
        return ProjectionHead(
            self.feat_dim, 2048, simclr_cfg.projection_dim, num_layers=2
        )

    def training_step(self, batch, batch_idx):
        views, labels = batch  # views: [2, B, C, H, W]
        h_i = self.backbone(views[0])  # representations
        h_j = self.backbone(views[1])
        z_i = self.projector(h_i)      # projections
        z_j = self.projector(h_j)
        loss = self.loss_fn(z_i, z_j)  # symmetric NT-Xent, no queue
        self.log_train_metrics(loss)
        return loss
```

### Pattern 2: Subclass Override for v2
**What:** SimCLRv2Module subclasses SimCLRv1Module and overrides only `build_projector()`.
**When to use:** Per D-02 -- v2 differs only in projection head depth.
**Example:**
```python
class SimCLRv2Module(SimCLRv1Module):
    def build_projector(self) -> nn.Module:
        simclr_cfg = self.cfg.simclr or SimCLRConfig()
        return ProjectionHead(
            self.feat_dim, 2048, simclr_cfg.projection_dim, num_layers=3
        )
```

### Pattern 3: Registration (follow Phase 2 pattern)
**What:** `methods/simclr/__init__.py` registers both methods; `methods/__init__.py` imports the sub-package.
**Example:**
```python
# methods/simclr/__init__.py
from core.dispatcher import register_method
from methods.simclr.module import SimCLRv1Module, SimCLRv2Module

register_method("simclr_v1", SimCLRv1Module)
register_method("simclr_v2", SimCLRv2Module)
```

```python
# methods/__init__.py -- add this line:
import methods.simclr  # noqa: F401
```

### Anti-Patterns to Avoid
- **Creating a new NTXentLoss class:** D-01 explicitly forbids this. `InfoNCELoss._symmetric_loss` is already NT-Xent.
- **Making num_layers configurable via YAML for v2:** D-02 says the switch is controlled by the subclass, not config.
- **Pre-normalizing embeddings before passing to InfoNCELoss:** The loss already L2-normalizes internally (verified in `core/losses.py` line 56-57).
- **Using `strong=False` for SimCLR augmentation:** SimCLR requires strong augmentation (s=1.0). The `SSLDataModule` defaults to `strong=True`, which is correct.
- **Hard-coding feature dimensions:** Use `backbone.num_features` via `build_backbone()` return value.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NT-Xent loss | Custom loss class | `InfoNCELoss(queue=None)` | Already implements symmetric mode with L2-normalization, masking, and cross-entropy (D-01) |
| Projection MLP | Manual nn.Sequential | `ProjectionHead(num_layers=N)` | Handles BN+ReLU on intermediates, BN-only on final layer |
| Strong augmentation | Custom transform pipeline | `ContrastiveAugmentation(strong=True)` | Already s=1.0 color jitter + Gaussian blur |
| LARS optimizer | External library | `core/optimizers.py::LARS` | Tutorial-readable from-scratch implementation already exists |
| LR scheduling | Manual scheduler setup | `BaseSSLModule.configure_optimizers()` | Warmup-cosine already wired at step granularity |
| Method registration | Manual dispatcher edits | `register_method()` | Registry pattern established in Phase 1 |

**Key insight:** Phase 1 was designed specifically to make Phase 3 (SimCLR) trivial. Every component SimCLR needs already exists. The implementation work is pure assembly and testing.

## Common Pitfalls

### Pitfall 1: Color Jitter Strength s=1.0
**What goes wrong:** Using default torchvision color jitter (s~0.4) instead of SimCLR's strong augmentation (s=1.0).
**Why it happens:** Default torchvision ColorJitter parameters are much weaker than SimCLR requires.
**How to avoid:** `ContrastiveAugmentation(strong=True)` already handles this. The YAML configs should use `strong=True` (which is the SSLDataModule default). Add a comment in the module docstring confirming s=1.0.
**Warning signs:** Training converges but representations are poor; augmented views look too similar.

### Pitfall 2: Batch Size Below 256
**What goes wrong:** SimCLR performance degrades sharply below batch_size=256 because effective negatives = 2*(batch_size-1).
**Why it happens:** In-batch contrastive methods need many negatives for discriminative representations.
**How to avoid:** Document in YAML config comments. Default batch_size=256 minimum. LARS enables larger batches (1024+).
**Warning signs:** Loss converges but downstream accuracy is poor.

### Pitfall 3: LARS Excluding BN/Bias Parameters
**What goes wrong:** Applying LARS trust ratio to batch norm and bias parameters destabilizes training.
**Why it happens:** BN/bias parameters have different gradient dynamics than weight tensors.
**How to avoid:** The existing `LARS` implementation already excludes 1-D parameters (bias, BN) from trust ratio scaling via `exclude_bias_and_norm=True` default.
**Warning signs:** Training diverges with LARS optimizer.

### Pitfall 4: SimCLRv2 Weight Decay Sensitivity
**What goes wrong:** Applying v1 hyperparameters to v2 without adjusting weight decay for the larger projection head.
**Why it happens:** 3-layer projection head has significantly more parameters than 2-layer.
**How to avoid:** Document in v2 class docstring per D-02. Increase weight_decay for v2 configs.
**Warning signs:** v2 representations underperform v1 despite deeper projection head.

### Pitfall 5: Forgetting to Return Both h and z
**What goes wrong:** Evaluation uses projection output `z` instead of backbone representation `h`.
**Why it happens:** Loss is computed on `z`, but downstream evaluation should use `h`.
**How to avoid:** The module computes `h = backbone(x)` and `z = projector(h)` separately. Loss uses `z`. For downstream eval, extract `h` from the backbone directly. Document this in the docstring.
**Warning signs:** Poor downstream accuracy despite good pretraining loss.

### Pitfall 6: Registry Collision in Tests
**What goes wrong:** Tests fail because method names are already registered from a previous test.
**Why it happens:** `register_method()` raises ValueError on duplicate names; the global registry persists across tests.
**How to avoid:** Use `clean_registry` fixture pattern from `test_invariant_spread.py` that saves/restores `_METHOD_REGISTRY`.
**Warning signs:** `ValueError: Method 'simclr_v1' is already registered` in test output.

## Code Examples

### NT-Xent Symmetry Test
```python
# Verified: InfoNCELoss._symmetric_loss already has this property.
# Existing test_losses.py::test_symmetric_symmetry_property covers this.
# Phase 3 should add an explicit test in test_simclr.py that uses the
# SimCLR-standard temperature (0.5) and verifies loss(z1, z2) == loss(z2, z1).
def test_ntxent_symmetry():
    loss_fn = InfoNCELoss(temperature=0.5)
    z1 = F.normalize(torch.randn(32, 128), dim=1)
    z2 = F.normalize(torch.randn(32, 128), dim=1)
    assert torch.isclose(loss_fn(z1, z2), loss_fn(z2, z1), atol=1e-5)
```

### Identical Views Minimum Loss Test
```python
# Verify that identical views produce the minimum possible loss.
def test_identical_views_minimum():
    loss_fn = InfoNCELoss(temperature=0.5)
    z = F.normalize(torch.randn(32, 128), dim=1)
    loss_identical = loss_fn(z, z.clone())
    loss_random = loss_fn(z, F.normalize(torch.randn(32, 128), dim=1))
    assert loss_identical < loss_random
```

### YAML Config Pattern
```yaml
# configs/simclr_v1_resnet18.yaml
# SimCLR v1 (Chen et al., ICML 2020)
# Note: SimCLR performance degrades sharply below batch_size=256
method: simclr_v1
backbone: resnet18
pretrained: false
max_epochs: 200
warmup_epochs: 10
batch_size: 256
lr: 0.3
weight_decay: 1e-6
optimizer: adamw
n_views: 2
data_dir: data
num_workers: 4

simclr:
  temperature: 0.5
  projection_dim: 128
```

### Augmentation Visualization Script Pattern
```python
# tools/visualize_augmentations.py
# Standalone CLI: python tools/visualize_augmentations.py
# Saves grid of 8 augmented views from one image to confirm s=1.0 and blur.
import argparse
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from core.data import ContrastiveAugmentation, IMAGENET_MEAN, IMAGENET_STD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="tools/output/augmentation_grid.png")
    parser.add_argument("--n-views", type=int, default=8)
    args = parser.parse_args()
    # Load one image, apply augmentation n_views times, save grid
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Memory bank (Instance Discrimination) | In-batch contrastive (SimCLR) | 2020 (ICML) | Eliminated stale-key problem; simpler architecture |
| 2-layer projection head (v1) | 3-layer projection head (v2) | 2020 (NeurIPS) | Improved representation quality for larger models |
| SGD optimizer for contrastive learning | LARS for large-batch SSL | 2020 | Enables batch sizes >1024 without degradation |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in pyproject.toml) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/test_simclr.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ERA2-03.1 | SimCLRv1 trains 5 epochs without loss divergence | integration | `pytest tests/test_simclr.py::test_simclr_v1_train_5_epochs -x` | Wave 0 |
| ERA2-03.2 | Color jitter s=1.0 confirmed | unit | `pytest tests/test_simclr.py::test_strong_augmentation_s1 -x` | Wave 0 |
| ERA2-03.3 | NT-Xent symmetry: loss(z1,z2)==loss(z2,z1) | unit | `pytest tests/test_simclr.py::test_ntxent_symmetry -x` | Wave 0 |
| ERA2-03.4 | Identical views yield minimum loss | unit | `pytest tests/test_simclr.py::test_identical_views_minimum -x` | Wave 0 |
| ERA2-03.5 | LARS activates when optimizer=lars | unit | `pytest tests/test_simclr.py::test_lars_optimizer_activates -x` | Wave 0 |
| ERA2-04.1 | SimCLRv2 uses 3-layer projection head | unit | `pytest tests/test_simclr.py::test_simclr_v2_3layer_head -x` | Wave 0 |
| ERA2-04.2 | v2 changes only projection depth, nothing else | unit | `pytest tests/test_simclr.py::test_v2_only_changes_projector -x` | Wave 0 |
| SC-3 | Both selectable via method: simclr_v1/simclr_v2 in YAML | integration | `pytest tests/test_simclr.py::test_dispatcher_registration -x` | Wave 0 |
| SC-4 | Per-method YAML configs exist and load | integration | `pytest tests/test_simclr.py::test_yaml_config_loads -x` | Wave 0 |
| SC-5 | LARS default is AdamW, LARS activates with config | unit | `pytest tests/test_simclr.py::test_default_optimizer_is_adamw -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_simclr.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_simclr.py` -- all test cases listed above
- No new framework install needed -- pytest already configured
- No new conftest fixtures needed -- `conftest.py` already has `tmp_imagefolder`, `random_tensor`, `toy_config_dict`
- Reuse `LossTracker` callback pattern from `test_invariant_spread.py` for training tests
- Reuse `clean_registry` fixture pattern from `test_invariant_spread.py`

## Open Questions

1. **Augmentation visualization script output directory**
   - What we know: D-03 specifies `tools/visualize_augmentations.py`, specifics suggest saving to `tools/output/augmentation_grid.png`
   - What's unclear: Whether `tools/` should have an `__init__.py` (making it a package) or remain a plain scripts directory
   - Recommendation: Leave as plain scripts directory (no `__init__.py`). Scripts in `tools/` are standalone CLI utilities, not importable modules. Create `tools/output/` directory or use `--output` arg with default.

2. **YAML config naming convention**
   - What we know: Phase 2 uses `{method}_{backbone}.yaml` pattern (e.g., `invariant_spread_resnet18.yaml`)
   - What's unclear: How to name the LARS variant config
   - Recommendation: `simclr_v1_resnet18.yaml` (AdamW default), `simclr_v1_resnet50_lars.yaml` (LARS variant for documentation), `simclr_v2_resnet18.yaml`. Keep at least one LARS config to satisfy SC-5.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `core/losses.py`, `core/projection.py`, `core/base.py`, `core/config.py`, `core/data.py`, `core/optimizers.py`, `core/dispatcher.py`, `core/backbone.py` -- all verified directly
- Codebase inspection: `methods/invariant_spread/module.py`, `methods/invariant_spread/__init__.py` -- reference implementation pattern
- Codebase inspection: `tests/test_losses.py`, `tests/test_invariant_spread.py` -- test patterns
- Codebase inspection: `configs/invariant_spread_resnet18.yaml` -- YAML config pattern

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions D-01 through D-05 -- user-locked design choices
- REQUIREMENTS.md ERA2-03, ERA2-04 -- method specifications

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all components verified in codebase, nothing new to install
- Architecture: HIGH -- follows established Phase 2 pattern exactly, all decisions locked
- Pitfalls: HIGH -- derived from REQUIREMENTS.md gotchas + verified codebase behavior

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable -- no external dependencies, all internal code)
