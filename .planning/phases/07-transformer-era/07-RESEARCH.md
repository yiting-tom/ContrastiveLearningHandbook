# Phase 7: Transformer Era - Research

**Researched:** 2026-04-09
**Domain:** Vision Transformer SSL (MoCo v3, DINO, DINOv2 feature extraction)
**Confidence:** HIGH

## Summary

Phase 7 implements three transformer-era SSL methods: MoCo v3 (in-batch symmetric contrastive with ViT), DINO (student-teacher self-distillation with centering/sharpening), and a DINOv2 feature extraction tutorial. All three build on established project patterns (EMA from `core/ema.py`, `BaseSSLModule` from `core/base.py`, `ProjectionHead`/`PredictorHead` from `core/projection.py`, `MultiCropDataset` from `core/data.py`) and require only ViT-specific adaptations.

The codebase is mature with 6 completed phases and well-established patterns for method implementation, testing, config, and dispatcher registration. The key new concepts are: (1) ViT backbones via timm with patch projection freeze, (2) DINO's centering buffer and teacher-temperature warmup, (3) a standalone DINOv2 demo script outside the training loop. All dependencies (timm 1.0.19+, PyTorch Lightning 2.6.1) are already installed.

**Primary recommendation:** Follow existing method patterns exactly (config class, module file, `__init__.py` registration, YAML config, unit tests) -- the only novelty is ViT-specific logic (patch freeze, centering buffer, multi-crop teacher/student split).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `MoCoV3Config(_StrictBase)` as a new class in `core/config.py` -- separate from `MoCoConfig`. Fields: `temperature: float = 0.2`, `momentum: float = 0.99`, `predictor_hidden_dim: int = 4096`. No `queue_size`. Registered in `TrainConfig` as `moco_v3: Optional[MoCoV3Config] = None`.
- **D-02:** DINO centering vector via `self.register_buffer('center', torch.zeros(n_prototypes))`. Updated in `training_step` before cross-entropy loss using `self.center = 0.9 * self.center + 0.1 * teacher_output.mean(dim=0)`.
- **D-03:** `eval/dinov2_demo.py` loads via `timm.create_model('vit_small_patch14_dinov2', pretrained=True)`. No `torch.hub` or `facebookresearch/dinov2` repo.
- **D-04:** `dinov2_demo.py` accepts `--dataset cifar10|stl10|imagefolder` (argparse), defaulting to `cifar10`. Runs both zero-shot k-NN and linear probing.
- **D-05:** Existing `PredictorHead('standard', input_dim=256, hidden_dim=4096, output_dim=256)` covers MoCo v3. Update docstring only.
- **D-06:** DINO head: `ProjectionHead(feat_dim, 2048, 256, num_layers=3)` + L2-norm + `nn.Linear(256, 65536, bias=False)` prototype layer. Teacher is EMA copy, no predictor.
- **D-07:** Freeze `backbone.patch_embed.proj.weight` and `.bias` via `requires_grad_(False)` in `__init__`, not `setup()`.
- **D-08:** Use `build_backbone('vit_small_patch16_224', pretrained=False)` for MoCo v3 and DINO. DINOv2 demo uses `vit_small_patch14_dinov2` (pretrained).

### Claude's Discretion
- Exact centering momentum value (0.9 default from D-02, adjustable via `DINOConfig`)
- `DINOv2Tutorial` argparse flag names beyond `--dataset` and `--data-dir`
- YAML config defaults for `moco_v3` and `dino` (aside from those locked in REQUIREMENTS.md)
- Exact smoke-test epoch count (roadmap specifies 3 epochs)
- DOC-02 docstring wording for the patch-projection freeze gotcha

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within Phase 7 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ERA4-01 | MoCo v3: ViT backbone, in-batch symmetric loss, momentum m=0.99, prediction MLP, patch projection freeze, AdamW, gradient clipping | Codebase has `InfoNCELoss` (symmetric mode), `EMAUpdater`, `PredictorHead('standard')`, `build_backbone` supporting ViT. Gradient clipping via `Trainer(gradient_clip_val=...)`. |
| ERA4-02 | DINO: student/teacher ViT, centering (before loss), sharpening (temp warmup), multi-crop 2g+6l, output dim 65536, gradient clipping max_norm=3.0 | `MultiCropDataset` in `core/data.py` handles multi-crop. `DINOConfig` already defined in `core/config.py`. `EMAUpdater` for teacher EMA. Centering via `register_buffer`. |
| ERA4-03 | DINOv2: feature extraction only, k-NN + linear probing, register token note, "DINOv3 does not exist" | timm has `vit_small_patch14_dinov2.lvd142m`. Standalone script in `eval/dinov2_demo.py`. |
| INFRA-05 | PredictorHead shared by BYOL, SimSiam, MoCo v3, DINO | `PredictorHead` in `core/projection.py` already has 'standard' and 'bottleneck' variants. Docstring update only needed. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| timm | 1.0.19 (installed; 1.0.26 in requirements.txt) | ViT backbone (`vit_small_patch16_224`, `vit_small_patch14_dinov2.lvd142m`) | Already the backbone factory for all methods [VERIFIED: requirements.txt + pip list] |
| lightning | 2.6.1 | Training loop, `Trainer(gradient_clip_val=...)` for ViT gradient clipping | Already used by all methods [VERIFIED: requirements.txt] |
| torch | 2.11.0 | Core tensor ops, `register_buffer` for centering vector | Already installed [VERIFIED: requirements.txt] |
| torchvision | 0.26.0 | CIFAR-10/STL-10 datasets for DINOv2 demo | Already installed [VERIFIED: requirements.txt] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | 2.12.5 | `MoCoV3Config(_StrictBase)` config class | Already used for all configs [VERIFIED: requirements.txt] |
| scikit-learn | (check) | k-NN classifier in DINOv2 demo | May need to add to requirements if not present |

**Installation:** No new dependencies required. All libraries already in `requirements.txt`.

## Architecture Patterns

### Recommended Project Structure
```
methods/
  moco_v3/
    __init__.py          # register_method("moco_v3", MoCoV3Module)
    module.py            # MoCoV3Module(BaseSSLModule)
  dino/
    __init__.py          # register_method("dino", DINOModule)
    module.py            # DINOModule(BaseSSLModule)
eval/
  dinov2_demo.py         # Standalone script (not registered in dispatcher)
configs/
  moco_v3_vit_small.yaml
  dino_vit_small.yaml
tests/
  test_moco_v3.py
  test_dino.py
```

### Pattern 1: ViT Backbone with Patch Projection Freeze (MoCo v3)
**What:** ViT's `patch_embed.proj` is a Conv2d that projects image patches to embeddings. MoCo v3 freezes this layer for training stability.
**When to use:** MoCo v3 (mandatory per paper). Not needed for DINO.
**Example:**
```python
# Source: timm ViT architecture [VERIFIED: timm source code on GitHub]
# vit_small_patch16_224 has: backbone.patch_embed.proj (nn.Conv2d)
# patch_embed.proj.weight shape: [384, 3, 16, 16]
# backbone.num_features = 384

self.backbone, self.feat_dim = build_backbone('vit_small_patch16_224', pretrained=False)
# Freeze in __init__, NOT setup() (D-07)
self.backbone.patch_embed.proj.weight.requires_grad_(False)
self.backbone.patch_embed.proj.bias.requires_grad_(False)
```

### Pattern 2: MoCo v3 Symmetric In-Batch Loss (No Queue)
**What:** Two views produce queries and keys via online and momentum encoders. Symmetric InfoNCE over in-batch pairs only. Prediction MLP on online branch.
**Example:**
```python
# Source: MoCo v3 paper (Chen et al., ICCV 2021) [ASSUMED]
# q1 = predictor(projector(backbone(v1)))
# q2 = predictor(projector(backbone(v2)))
# k1 = projector_ema(backbone_ema(v1))  # no predictor on momentum
# k2 = projector_ema(backbone_ema(v2))  # no predictor on momentum
# loss = (InfoNCE(q1, k2) + InfoNCE(q2, k1)) / 2
```

### Pattern 3: DINO Cross-Entropy with Centering + Sharpening
**What:** Teacher outputs are centered (subtract running mean) and sharpened (low temperature). Student outputs use higher temperature. Loss is cross-entropy between teacher and student softmax distributions.
**Example:**
```python
# Source: DINO paper (Caron et al., ICCV 2021) [ASSUMED]
# In training_step, BEFORE computing loss:
self.center = 0.9 * self.center + 0.1 * teacher_out.mean(dim=0)

# Teacher: center then sharpen
teacher_probs = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)

# Student: normal temperature
student_probs = F.log_softmax(student_out / student_temp, dim=-1)

# Cross-entropy loss (teacher is target, student is prediction)
loss = -torch.sum(teacher_probs * student_probs, dim=-1).mean()
```

### Pattern 4: DINO Multi-Crop Teacher/Student Split
**What:** Teacher sees only global crops (views[:2]). Student sees all crops (views[:] = 2 global + 6 local). Loss computed over all (teacher_global, student_crop) pairs where student_crop != teacher_global.
**Example:**
```python
# Source: DINO paper + existing MultiCropDataset [VERIFIED: core/data.py]
# crops_list from ssl_collate_multi_crop: list of tensors
# crops_list[:2] = global crops (224x224)
# crops_list[2:] = local crops (96x96)
global_crops = crops_list[:2]   # teacher input
all_crops = crops_list          # student input
```

### Pattern 5: EMA Setup in setup() (BYOL Pattern)
**What:** EMA updater requires `total_steps` from trainer, which is only available in `setup('fit')`, not `__init__`.
**Example:**
```python
# Source: methods/byol/module.py [VERIFIED: codebase]
def setup(self, stage: str) -> None:
    if stage == "fit" and self.trainer is not None:
        total_steps = self.trainer.estimated_stepping_batches
        self.ema = EMAUpdater(
            base_momentum=0.99,
            end_momentum=0.99,  # constant for MoCo v3
            total_steps=int(total_steps),
        )
```

### Pattern 6: Gradient Clipping via Trainer
**What:** Gradient clipping is configured at the `Trainer` level, not in the module. DINO uses `max_norm=3.0` per the paper.
**Example:**
```python
# Source: .planning/research/STACK.md [VERIFIED: codebase research]
# In YAML smoke tests and training scripts:
trainer = L.Trainer(gradient_clip_val=3.0, gradient_clip_algorithm="norm")
# This works with automatic optimization (default for all methods)
```

### Anti-Patterns to Avoid
- **Freezing patch_embed in setup() instead of __init__:** The freeze must happen in `__init__` so it survives checkpoint resume. Doing it in `setup()` could be overridden by checkpoint loading.
- **Passing local crops to DINO teacher:** Teacher must only see global crops. Passing local crops changes the training objective semantics.
- **Using MoCoConfig for MoCo v3:** MoCo v3 has no queue, different temperature (0.2 vs 0.07), different momentum (0.99 vs 0.999). A separate `MoCoV3Config` is required (D-01).
- **Computing centering after the loss:** Centering must be updated BEFORE the loss computation (D-02, matching running_mean convention).
- **Adding predictor to momentum branch:** Both MoCo v3 and DINO have the predictor on the online/student branch only. Momentum/teacher has no predictor.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ViT backbone | Custom ViT implementation | `build_backbone('vit_small_patch16_224')` via timm | timm handles all attention, positional encoding, CLS token |
| EMA momentum updates | Manual parameter copy loop | `EMAUpdater` from `core/ema.py` | Handles cosine schedule, no-grad context, step counting |
| Multi-crop augmentation | Custom crop logic | `MultiCropDataset` from `core/data.py` | Already handles variable-size crops with proper collation |
| Projection MLP | Ad-hoc `nn.Sequential` | `ProjectionHead` from `core/projection.py` | Consistent BN+ReLU convention across all methods |
| Prediction MLP | New predictor class | `PredictorHead('standard')` from `core/projection.py` | Already covers BYOL/SimSiam/MoCo v3 architecture |
| k-NN evaluation | Custom k-NN loop | `sklearn.neighbors.KNeighborsClassifier` | Robust, handles edge cases |

**Key insight:** Phase 7 reuses nearly all existing infrastructure. The only truly new code is: (1) MoCo v3 loss logic (symmetric in-batch without queue), (2) DINO centering/sharpening/teacher-student split, (3) DINOv2 demo script.

## Common Pitfalls

### Pitfall 1: ViT Patch Projection Not Frozen
**What goes wrong:** MoCo v3 with ViT produces poor representations silently -- no training crash, just bad downstream accuracy.
**Why it happens:** The patch projection layer is essentially a low-level feature extractor. Without freezing, ViT self-attention destabilizes early training.
**How to avoid:** Freeze `backbone.patch_embed.proj.weight` and `.bias` immediately in `__init__` (D-07). Unit test asserts `requires_grad == False`.
**Warning signs:** No obvious signal during training -- only visible via evaluation.

### Pitfall 2: DINO Centering Update Order
**What goes wrong:** If centering is updated AFTER loss computation, the running mean lags by one step, causing early instability.
**Why it happens:** Natural instinct is to compute loss first, then update statistics.
**How to avoid:** Update `self.center` BEFORE computing the cross-entropy loss in `training_step` (D-02).
**Warning signs:** Early training instability, NaN losses in first few epochs.

### Pitfall 3: DINO Teacher Temperature Warmup
**What goes wrong:** Starting teacher temperature at the final value (0.07) causes early training instability.
**Why it happens:** High initial teacher temperature produces uniform soft labels that don't provide useful signal.
**How to avoid:** Warmup teacher_temp from 0.04 to 0.07 over `warmup_teacher_temp_epochs` (30 epochs default in `DINOConfig`).
**Warning signs:** Loss oscillation or NaN in first 5 epochs.

### Pitfall 4: MoCo v3 Momentum Value
**What goes wrong:** Using m=0.999 (MoCo v1/v2 default) instead of m=0.99 causes 1-2% accuracy drop.
**Why it happens:** Copy-paste from MoCo v1/v2 config.
**How to avoid:** `MoCoV3Config` has separate `momentum=0.99` default (D-01). Never reuse `MoCoConfig`.
**Warning signs:** Subtle -- only visible in final accuracy numbers.

### Pitfall 5: DINOv2 timm Model Name
**What goes wrong:** `timm.create_model('vit_small_patch14_dinov2', pretrained=True)` may not resolve correctly -- the full qualified name is `vit_small_patch14_dinov2.lvd142m`.
**Why it happens:** timm model naming convention uses `.` suffixes for pretrained weight variants.
**How to avoid:** Use the full name `vit_small_patch14_dinov2.lvd142m` or verify that the short name resolves. Test in the demo script. [VERIFIED: Hugging Face model page shows `vit_small_patch14_dinov2.lvd142m`]
**Warning signs:** `RuntimeError` on model creation.

### Pitfall 6: Gradient Clipping Not in TrainConfig
**What goes wrong:** Gradient clipping is passed to `Trainer`, not stored in `TrainConfig`. YAML configs cannot specify it.
**Why it happens:** The current `TrainConfig` schema has no `gradient_clip_val` field.
**How to avoid:** Either add a `gradient_clip_val: Optional[float] = None` field to `TrainConfig`, or document in YAML comments that gradient clipping should be configured at the Trainer level. The ROADMAP specifies `Trainer(gradient_clip_val=...)` for MoCo v3.
**Warning signs:** ViT training diverges without gradient clipping.

## Code Examples

### MoCoV3Config (new in core/config.py)
```python
# Source: D-01 from CONTEXT.md [VERIFIED: user decision]
class MoCoV3Config(_StrictBase):
    """MoCo v3 (Chen et al., ICCV 2021) method-specific hyper-parameters.
    
    Note: temperature=0.2 differs from MoCo v1/v2's 0.07 (per paper).
    Note: momentum=0.99 differs from v1/v2's 0.999 (per paper).
    Note: No queue_size -- MoCo v3 uses in-batch keys only.
    """
    temperature: float = 0.2
    momentum: float = 0.99
    predictor_hidden_dim: int = 4096
```

### MoCoV3Module training_step skeleton
```python
# Source: MoCo v3 paper + existing MoCo patterns [VERIFIED: methods/moco/module.py]
def training_step(self, batch, batch_idx):
    views, labels = batch  # views shape: [2, B, C, H, W]
    v1, v2 = views[0], views[1]
    
    # Online branch (with predictor)
    z1 = self.projector(self.backbone(v1))
    z2 = self.projector(self.backbone(v2))
    q1 = self.predictor(z1)
    q2 = self.predictor(z2)
    
    # Momentum branch (no predictor, no gradient)
    with torch.no_grad():
        k1 = self.projector_ema(self.backbone_ema(v1))
        k2 = self.projector_ema(self.backbone_ema(v2))
    
    # Symmetric InfoNCE (in-batch, no queue)
    loss = (self.loss_fn(q1, k2) + self.loss_fn(q2, k1)) / 2
    
    self.log_train_metrics(loss)
    return loss
```

### DINOModule centering + sharpening
```python
# Source: DINO paper + D-02, D-06 [VERIFIED: user decisions]
def __init__(self, cfg):
    # ...
    self.register_buffer('center', torch.zeros(n_prototypes))
    
def training_step(self, batch, batch_idx):
    crops_list, labels = batch
    global_crops = crops_list[:2]   # teacher sees only these
    all_crops = crops_list          # student sees all
    
    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_outs = []
        for crop in global_crops:
            h = self.backbone_ema(crop)
            z = self.projector_ema(h)
            z = F.normalize(z, dim=-1)
            logits = self.prototype_layer_ema(z)  # [B, 65536]
            teacher_outs.append(logits)
        teacher_out = torch.cat(teacher_outs, dim=0)  # [2B, 65536]
        
        # Update center BEFORE loss (D-02)
        self.center = 0.9 * self.center + 0.1 * teacher_out.mean(dim=0)
        
        # Centering + sharpening
        teacher_probs = F.softmax(
            (teacher_out - self.center) / self.teacher_temp, dim=-1
        )
    
    # Student forward (gradient flows)
    total_loss = 0
    n_loss_terms = 0
    for i, crop in enumerate(all_crops):
        h = self.backbone(crop)
        z = self.projector(h)
        z = F.normalize(z, dim=-1)
        student_logits = self.prototype_layer(z)  # [B, 65536]
        student_log_probs = F.log_softmax(student_logits / self.student_temp, dim=-1)
        
        # Cross-entropy with each global teacher output
        for j, t_prob in enumerate(teacher_probs.chunk(2)):
            if i == j:
                continue  # skip same-view pairs
            total_loss += -torch.sum(t_prob * student_log_probs, dim=-1).mean()
            n_loss_terms += 1
    
    loss = total_loss / n_loss_terms
    self.log_train_metrics(loss)
    return loss
```

### DINOv2 demo k-NN evaluation skeleton
```python
# Source: D-03, D-04 [VERIFIED: user decisions]
# eval/dinov2_demo.py (standalone script)
import argparse
import timm
import torch
from torchvision.datasets import CIFAR10
from sklearn.neighbors import KNeighborsClassifier

def extract_features(model, dataloader, device):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            feats = model(imgs.to(device))
            features.append(feats.cpu())
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'stl10', 'imagefolder'], default='cifar10')
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()
    
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
    # ... feature extraction, k-NN, linear probing
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Queue-based contrastive (MoCo v1/v2) | In-batch symmetric (MoCo v3) | ICCV 2021 | Queue removed; larger batches sufficient with ViT |
| Fixed temperature teacher | Centering + sharpening (DINO) | ICCV 2021 | Prevents mode collapse without negatives |
| ResNet backbones | ViT backbones | 2021+ | Higher downstream accuracy; requires patch freeze trick |
| Training DINOv2 from scratch | Feature extraction only | 2023 | Pre-training requires hundreds of GPU-days; features are universal |
| `timm` short model names | `timm` qualified names with `.` suffix | timm 1.0+ | `vit_small_patch14_dinov2.lvd142m` not just `vit_small_patch14_dinov2` |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MoCo v3 loss is symmetric InfoNCE computed as `(loss(q1,k2) + loss(q2,k1)) / 2` using the existing `InfoNCELoss` | Architecture Patterns | Loss function may need modification for in-batch-only mode without queue arg |
| A2 | DINO student temperature is higher than teacher temperature (e.g., 0.1 vs 0.04-0.07) | Code Examples | Incorrect temperature ratio would change training dynamics |
| A3 | `vit_small_patch14_dinov2` short name resolves in timm to `vit_small_patch14_dinov2.lvd142m` | Pitfall 5 | Script may error; use full qualified name as fallback |
| A4 | scikit-learn is available in the project environment for k-NN evaluation | Standard Stack | May need `pip install scikit-learn` |

## Open Questions

1. **Gradient clipping in TrainConfig?**
   - What we know: `Trainer(gradient_clip_val=...)` works, but current `TrainConfig` has no field for it. YAML configs cannot specify gradient clipping.
   - What's unclear: Should we add `gradient_clip_val: Optional[float] = None` to `TrainConfig` or handle it only at the Trainer level in test/training scripts?
   - Recommendation: Add the field to `TrainConfig` so YAML configs can specify it -- both MoCo v3 and DINO need it, and future methods may too. This is a small config schema extension.

2. **InfoNCELoss in-batch-only mode**
   - What we know: Existing `InfoNCELoss` accepts an optional `queue` parameter. When `queue=None`, it operates on in-batch pairs.
   - What's unclear: Whether the existing InfoNCELoss correctly handles the symmetric in-batch case that MoCo v3 needs (query-key cross-correlation without self-pairs).
   - Recommendation: Verify the existing `InfoNCELoss(q, k, queue=None)` path works for MoCo v3. If not, a small adapter or mode parameter may be needed.

3. **DINOv2 demo timm model name resolution**
   - What we know: HuggingFace lists `vit_small_patch14_dinov2.lvd142m` as the full name.
   - What's unclear: Whether `timm.create_model('vit_small_patch14_dinov2', pretrained=True)` auto-resolves to the `.lvd142m` variant.
   - Recommendation: Use the full qualified name `vit_small_patch14_dinov2.lvd142m` in the demo script to be explicit.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.1 |
| Config file | `tests/conftest.py` (shared fixtures) |
| Quick run command | `python -m pytest tests/test_moco_v3.py tests/test_dino.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ERA4-01a | `patch_embed.proj.weight.requires_grad == False` | unit | `pytest tests/test_moco_v3.py::test_patch_projection_frozen -x` | Wave 0 |
| ERA4-01b | MoCo v3 trains 3 epochs, loss finite | smoke | `pytest tests/test_moco_v3.py::test_moco_v3_train_3_epochs -x` | Wave 0 |
| ERA4-01c | MoCo v3 uses AdamW optimizer | unit | `pytest tests/test_moco_v3.py::test_moco_v3_uses_adamw -x` | Wave 0 |
| ERA4-02a | Centering buffer updated before loss | unit | `pytest tests/test_dino.py::test_centering_update_before_loss -x` | Wave 0 |
| ERA4-02b | Teacher receives only global crops | unit | `pytest tests/test_dino.py::test_teacher_global_crops_only -x` | Wave 0 |
| ERA4-02c | DINO trains 3 epochs, loss finite | smoke | `pytest tests/test_dino.py::test_dino_train_3_epochs -x` | Wave 0 |
| ERA4-03 | DINOv2 demo loads model, runs k-NN | smoke | `pytest tests/test_dinov2_demo.py -x` | Wave 0 |
| INFRA-05 | PredictorHead docstring lists BYOL, SimSiam, MoCo v3, DINO | unit | `pytest tests/test_predictor_head.py::test_predictor_docstring -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_moco_v3.py tests/test_dino.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_moco_v3.py` -- covers ERA4-01 (patch freeze, training, AdamW, dispatcher)
- [ ] `tests/test_dino.py` -- covers ERA4-02 (centering, global crops, training, dispatcher)
- [ ] `tests/test_dinov2_demo.py` -- covers ERA4-03 (model loading, feature extraction)
- [ ] Existing `tests/test_predictor_head.py` needs docstring assertion update for INFRA-05

## Security Domain

Not applicable. This phase implements ML training algorithms and a demo script -- no authentication, user input handling, cryptography, or network services.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `core/config.py` (DINOConfig exists, MoCoV3Config needs creation), `core/projection.py` (PredictorHead covers MoCo v3), `core/ema.py` (EMAUpdater), `core/data.py` (MultiCropDataset), `core/base.py` (BaseSSLModule), `core/backbone.py` (build_backbone), `core/dispatcher.py` (register_method pattern)
- Codebase inspection: `methods/byol/module.py` (EMA pattern reference), `methods/moco/module.py` (MoCo v1/v2 pattern reference), `methods/swav/module.py` (multi-crop pattern reference)
- [timm vit_small_patch14_dinov2.lvd142m HuggingFace](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) -- model name verification
- [timm PatchEmbed source](https://github.com/pprp/timm/blob/master/timm/layers/patch_embed.py) -- `patch_embed.proj` is Conv2d
- `requirements.txt` -- verified timm 1.0.26, torch 2.11.0, lightning 2.6.1, pydantic 2.12.5

### Secondary (MEDIUM confidence)
- [timm vit_small_patch16_224.augreg_in21k HuggingFace](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k) -- ViT-Small has 384 features
- `.planning/research/STACK.md` -- gradient clipping via Trainer documentation

### Tertiary (LOW confidence)
- MoCo v3 / DINO paper details (training dynamics, loss formulation) -- from training knowledge, not verified against original papers in this session

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and used in prior phases
- Architecture: HIGH -- patterns established in 6 prior phases; ViT specifics verified via timm docs
- Pitfalls: HIGH -- derived from paper-specific gotchas documented in REQUIREMENTS.md
- DINOv2 demo: MEDIUM -- timm model name needs runtime verification

**Research date:** 2026-04-09
**Valid until:** 2026-05-09 (stable -- all dependencies pinned in requirements.txt)
