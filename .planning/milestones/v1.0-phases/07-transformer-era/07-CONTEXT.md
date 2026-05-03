# Phase 7: Transformer Era - Context

**Gathered:** 2026-04-09 (discuss mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

MoCo v3, DINO, and DINOv2 (feature extraction only) are implemented — demonstrating how the contrastive/self-distillation paradigm transfers to Vision Transformers. Includes the patch-projection freeze trick (MoCo v3), centering + sharpening (DINO), and a standalone DINOv2 demo script using pretrained weights. All three methods are selectable via YAML. No from-scratch DINOv2 training (requires hundreds of GPU-days).

</domain>

<decisions>
## Implementation Decisions

### MoCo v3 Config

- **D-01:** Create `MoCoV3Config(_StrictBase)` as a **new class** in `core/config.py` — separate from `MoCoConfig`. Fields: `temperature: float = 0.2`, `momentum: float = 0.99`, `predictor_hidden_dim: int = 4096`. No `queue_size` (MoCo v3 has no queue). Registered in `TrainConfig` as `moco_v3: Optional[MoCoV3Config] = None`. `MoCoConfig` remains unchanged for v1/v2.

### DINO Centering Vector

- **D-02:** Store the DINO centering vector via `self.register_buffer('center', torch.zeros(n_prototypes))` — participates in `state_dict` and checkpoint/resume. Updated in `training_step` **before** the cross-entropy loss using a momentum update: `self.center = 0.9 * self.center + 0.1 * teacher_output.mean(dim=0)`. This matches the BatchNorm `running_mean` convention and prevents centering reset on checkpoint resume.

### DINOv2 Tutorial Loading

- **D-03:** `eval/dinov2_demo.py` loads the model via `timm.create_model('vit_small_patch14_dinov2', pretrained=True)`. Stays within the existing `timm` dependency — no `torch.hub` or `facebookresearch/dinov2` repo required. Docstring must note: (1) register tokens added Oct 2023 change the model API; (2) the correct lineage is DINO → DINOv2 → DINOv2+Registers; (3) "DINOv3" does not exist.

### DINOv2 Tutorial Dataset

- **D-04:** `dinov2_demo.py` accepts `--dataset cifar10|stl10|imagefolder` (argparse), defaulting to `cifar10`. CIFAR-10 uses `torchvision.datasets.CIFAR10(root=..., download=True)` — auto-downloads, zero setup. Runs both zero-shot k-NN and linear probing. For `imagefolder`, accept `--data-dir <path>` pointing to an ImageFolder-compatible directory.

### PredictorHead Coverage for MoCo v3

- **D-05:** Verify that existing `PredictorHead('standard', input_dim=256, hidden_dim=4096, output_dim=256)` covers MoCo v3's prediction MLP (2-layer MLP on top of projection output). No new variant needed — `'standard'` is exactly the right architecture. Update `PredictorHead` docstring to list BYOL, SimSiam, MoCo v3, and DINO student head as consumers. `PredictorHead` is instantiated on the **online branch only** — momentum encoder has no predictor.

### DINO Head Architecture

- **D-06:** The DINO student/teacher head is: `ProjectionHead(feat_dim, hidden_dim=2048, output_dim=256, num_layers=3)` for the MLP, followed by L2-normalization, followed by `nn.Linear(256, 65536, bias=False)` as the final prototype layer (not weight-normalized — tutorial simplification). Teacher head is an EMA copy (same structure, no predictor). This reuses the existing `ProjectionHead` without code duplication.

### MoCo v3 Patch Projection Freeze

- **D-07:** Freeze `backbone.patch_embed.proj` (weight + bias) immediately after model construction: `self.backbone.patch_embed.proj.weight.requires_grad_(False)` and `self.backbone.patch_embed.proj.bias.requires_grad_(False)`. This must happen in `__init__`, not `setup()`. Verified by the unit test: `assert backbone.patch_embed.proj.weight.requires_grad == False`.

### MoCo v3 Backbone

- **D-08:** Use `build_backbone('vit_small_patch16_224', pretrained=False)` via `timm` for MoCo v3. DINO also uses `vit_small_patch16_224`. DINOv2 demo uses `vit_small_patch14_dinov2` (pretrained). All three resolve through timm — no new backbone factory needed.

### Claude's Discretion

- Exact centering momentum value (0.9 default from D-02, adjustable via `DINOConfig`)
- `DINOv2Tutorial` argparse flag names beyond `--dataset` and `--data-dir`
- YAML config defaults for `moco_v3` and `dino` (aside from those locked in REQUIREMENTS.md)
- Exact smoke-test epoch count (roadmap specifies 3 epochs)
- DOC-02 docstring wording for the patch-projection freeze gotcha

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Method specifications (primary)
- `.planning/REQUIREMENTS.md` §ERA4-01 — MoCo v3 full spec: ViT backbone, in-batch symmetric loss, momentum m=0.99, prediction MLP, patch projection freeze, AdamW, gradient clipping
- `.planning/REQUIREMENTS.md` §ERA4-02 — DINO full spec: student/teacher ViT, centering (update before loss), sharpening (temperature warmup), multi-crop 2 global + 6 local, output dim 65536, gradient clipping max_norm=3.0
- `.planning/REQUIREMENTS.md` §ERA4-03 — DINOv2 spec: feature extraction only, zero-shot k-NN + linear probing, register token API note, "DINOv3 does not exist" clarification

### Phase roadmap
- `.planning/ROADMAP.md` §Phase 7 — Goal, 5 success criteria, 8 pre-specified plan outlines (07-01 through 07-08)

### Foundation codebase
- `core/config.py` — `DINOConfig` already defined (n_prototypes=65536, teacher_temp, warmup fields); add `MoCoV3Config` here; `TrainConfig` for new `moco_v3` field
- `core/projection.py` — `PredictorHead` ('standard' covers MoCo v3); `ProjectionHead` (reused for DINO MLP portion)
- `core/ema.py` — `EMAUpdater` reused for both MoCo v3 (momentum encoder) and DINO teacher EMA
- `core/data.py` — `MultiCropDataset` (built in Phase 5) — DINO uses 2 global + 6 local crops
- `core/base.py` — `BaseSSLModule.on_train_batch_end` hook for EMA updates
- `core/dispatcher.py` — `register_method()` for `moco_v3` and `dino` registration

### Reference implementations
- `methods/byol/module.py` — reference for student/teacher EMA pattern (`backbone_ema`, `projector_ema`, `on_train_batch_end` EMA call)
- `methods/moco/module.py` — reference for MoCo EMA pattern (MoCo v3 extends this paradigm)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PredictorHead('standard', input_dim=256, hidden_dim=4096, output_dim=256)` — covers MoCo v3 prediction MLP without modification; docstring update needed
- `ProjectionHead(feat_dim, 2048, 256, num_layers=3)` — reused for DINO's MLP portion (backbone → 256-dim)
- `EMAUpdater(base_momentum, end_momentum, total_steps)` — used identically for DINO teacher; call `step(student_params, teacher_params)` in `on_train_batch_end`
- `MultiCropDataset` in `core/data.py` — already handles 2-global + N-local crops; DINO configures via `n_large_crops=2, large_size=224, n_small_crops=6, small_size=96`
- `build_backbone('vit_small_patch16_224', pretrained=False)` — returns `(backbone, feat_dim)` via timm; patch_embed.proj is accessible for freeze

### Established Patterns
- Method packages in `methods/{method}/module.py` + `methods/{method}/__init__.py` with `register_method()` call
- Config: `cfg.moco_v3 or MoCoV3Config()` fallback pattern (consistent with other methods)
- EMA in `__init__`: `copy.deepcopy(backbone)`, freeze with `requires_grad_(False)`, then assign `self.ema_updater` + set `self._online_params` / `self._target_params` (or call `ema.step()` directly in `on_train_batch_end`)
- `learnable_params` property excludes momentum encoder params
- DOC-02 docstrings: paper title, authors, venue, arXiv link, gotcha list

### Integration Points
- `methods/moco_v3/__init__.py` → `register_method("moco_v3", MoCoV3Module)`
- `methods/dino/__init__.py` → `register_method("dino", DINOModule)`
- `methods/__init__.py` → add imports for `moco_v3` and `dino` sub-packages
- `DINOModule.training_step` → `MultiCropDataset` output is a list of tensors; teacher receives only `views[:2]` (global crops), student receives all views
- `eval/dinov2_demo.py` → standalone script, not registered in dispatcher

</code_context>

<specifics>
## Specific Ideas

- MoCo v3: `temperature=0.2` default (per paper, differs from v1/v2's 0.07) — document this in `MoCoV3Config` docstring
- DINO centering: `m=0.9` for centering momentum — can be made configurable via `DINOConfig` extension
- DINOv2 demo: show both k-NN and linear probing as back-to-back evaluations in the same script, not separate scripts
- Patch projection freeze gotcha: document clearly — "without freezing patch_embed.proj, ViT training silently degrades representations"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 7 scope.

</deferred>

---

*Phase: 07-transformer-era*
*Context gathered: 2026-04-09*
