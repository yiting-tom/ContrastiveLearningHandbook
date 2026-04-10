"""Pydantic v2 configuration schema for the contrastive learning tutorial repo.

All config classes use ``extra='forbid'`` so that unknown YAML keys raise a
``ValidationError`` immediately — a deliberate design choice (D-08) to help
tutorial users catch copy-paste typos early.

Usage::

    from core.config import TrainConfig, load_config

    cfg = load_config("configs/example.yaml")
    print(cfg.method, cfg.simclr.temperature)
"""
from __future__ import annotations

from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Base — ALL config classes inherit this to get extra='forbid'
# ---------------------------------------------------------------------------

class _StrictBase(BaseModel):
    """Shared base with ``extra='forbid'`` so unknown keys raise ValidationError."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Per-method sub-configs
# ---------------------------------------------------------------------------

class SimCLRConfig(_StrictBase):
    """SimCLR / SimCLR v2 method-specific hyper-parameters."""

    temperature: float = 0.5
    projection_dim: int = 128


class MoCoConfig(_StrictBase):
    """MoCo (v1/v2/v3) method-specific hyper-parameters."""

    temperature: float = 0.07
    queue_size: int = 65536
    momentum: float = 0.999


class BYOLConfig(_StrictBase):
    """BYOL method-specific hyper-parameters."""

    base_momentum: float = 0.996
    end_momentum: float = 1.0


class SwAVConfig(_StrictBase):
    """SwAV method-specific hyper-parameters."""

    n_prototypes: int = 3000
    freeze_prototypes_epochs: int = 1
    sinkhorn_iterations: int = 3
    temperature: float = 0.1
    epsilon: float = 0.05
    n_large_crops: int = 2
    large_size: int = 224
    n_small_crops: int = 6
    small_size: int = 96


class BarlowTwinsConfig(_StrictBase):
    """Barlow Twins method-specific hyper-parameters."""

    lambda_coeff: float = 5e-3
    projection_dim: int = 8192


class SimSiamConfig(_StrictBase):
    """SimSiam method-specific hyper-parameters."""

    predictor_hidden_dim: int = 512


class MoCoV3Config(_StrictBase):
    """MoCo v3 (Chen et al., ICCV 2021) method-specific hyper-parameters.

    Differs from MoCo v1/v2 in three key ways:
    - temperature=0.2 differs from MoCo v1/v2's 0.07 (per paper): higher
      temperature improves stability with ViT backbones.
    - momentum=0.99 differs from v1/v2's 0.999 (per paper): slower EMA gives
      better representations with large-batch ViT training.
    - No queue_size -- MoCo v3 uses in-batch keys only.
    """

    temperature: float = 0.2
    momentum: float = 0.99
    predictor_hidden_dim: int = 4096


class DINOConfig(_StrictBase):
    """DINO method-specific hyper-parameters."""

    n_prototypes: int = 65536
    teacher_temp: float = 0.04
    warmup_teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30
    student_temp: float = 0.1
    centering_momentum: float = 0.9


class SupConConfig(_StrictBase):
    """Supervised Contrastive Learning method-specific hyper-parameters."""

    temperature: float = 0.07
    n_samples_per_class: int = 2
    n_classes_per_batch: int = 8    # classes sampled per batch by ClassBalancedSampler
    num_classes: int = 10           # number of output classes (for stage-2 head)
    projection_dim: int = 128       # projection head output dimension


class InstanceDiscriminationConfig(_StrictBase):
    """Instance Discrimination (Wu et al., CVPR 2018) method-specific hyper-parameters."""

    temperature: float = 0.07
    n_negatives: int = 4096
    projection_dim: int = 128


class InvariantSpreadConfig(_StrictBase):
    """Invariant Spread (Ye et al., CVPR 2019) method-specific hyper-parameters."""

    temperature: float = 0.07
    projection_dim: int = 128


class InfoMinConfig(_StrictBase):
    """InfoMin (Tian et al., NeurIPS 2020) method-specific hyper-parameters.

    Controls augmentation policy: aggressive color jitter, higher grayscale
    probability, and no Gaussian blur (the InfoMin key difference vs SimCLR).
    """

    color_strength: float = 1.5
    grayscale_prob: float = 0.4
    use_blur: bool = False


# ---------------------------------------------------------------------------
# Eval sub-configs
# ---------------------------------------------------------------------------

class LinearProbeConfig(_StrictBase):
    """Configuration for linear probing evaluation."""

    max_epochs: int = 100
    lr: float = 0.1
    milestones: list[int] = [60, 80]


class KNNConfig(_StrictBase):
    """Configuration for k-NN evaluation."""

    k: int = 200
    temperature: float = 0.07
    every_n_epochs: int = 5


class TSNEConfig(_StrictBase):
    """Configuration for t-SNE visualization."""

    n_samples: int = 2000
    perplexities: list[int] = [10, 30, 50]


class UMAPConfig(_StrictBase):
    """Configuration for UMAP visualization."""

    n_samples: int = 5000
    metric: str = "cosine"


class FinetuneConfig(_StrictBase):
    """Configuration for fine-tuning evaluation."""

    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    freeze_bn: bool = True


class CAMConfig(_StrictBase):
    """Configuration for Class Activation Map visualization."""

    method: str = "eigencam"
    n_images: int = 8


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------

class EvalConfig(_StrictBase):
    """Top-level eval block — all sub-schemas are Optional."""

    linear_probe: Optional[LinearProbeConfig] = None
    knn: Optional[KNNConfig] = None
    tsne: Optional[TSNEConfig] = None
    umap: Optional[UMAPConfig] = None
    finetune: Optional[FinetuneConfig] = None
    cam: Optional[CAMConfig] = None


# ---------------------------------------------------------------------------
# TrainConfig — top-level schema
# ---------------------------------------------------------------------------

class TrainConfig(_StrictBase):
    """Top-level training configuration.

    Every YAML field maps directly to a field here.  Unknown keys raise a
    ``ValidationError`` (``extra='forbid'``).
    """

    # Required
    method: str

    # Backbone
    backbone: str = "resnet50"
    pretrained: bool = False

    # Training schedule
    max_epochs: int = 100
    warmup_epochs: int = 10
    batch_size: int = 256
    lr: float = 0.3
    weight_decay: float = 1e-6
    optimizer: Literal["adamw", "sgd", "lars"] = "adamw"
    scheduler: Literal["warmup_cosine"] = "warmup_cosine"
    gradient_clip_val: Optional[float] = None

    # Data
    n_views: int = 2
    data_dir: str = "data"
    num_workers: int = 4

    # Per-method sub-configs (all Optional, default None)
    simclr: Optional[SimCLRConfig] = None
    moco: Optional[MoCoConfig] = None
    moco_v3: Optional[MoCoV3Config] = None
    byol: Optional[BYOLConfig] = None
    swav: Optional[SwAVConfig] = None
    infomin: Optional[InfoMinConfig] = None
    barlow_twins: Optional[BarlowTwinsConfig] = None
    simsiam: Optional[SimSiamConfig] = None
    dino: Optional[DINOConfig] = None
    supcon: Optional[SupConConfig] = None
    instance_discrimination: Optional[InstanceDiscriminationConfig] = None
    invariant_spread: Optional[InvariantSpreadConfig] = None

    # Evaluation
    eval: Optional[EvalConfig] = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_config(path: str) -> TrainConfig:
    """Load and validate a YAML config file.

    Args:
        path: Path to a YAML file conforming to the ``TrainConfig`` schema.

    Returns:
        A fully-validated ``TrainConfig`` instance.

    Raises:
        pydantic.ValidationError: If the YAML contains unknown keys or invalid
            types at any nesting level.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return TrainConfig.model_validate(raw)
