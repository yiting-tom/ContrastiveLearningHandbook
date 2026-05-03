"""InfoMin (Tian et al., NeurIPS 2020).

Augmentation-policy demonstration of the InfoMin principle: views should
share task-relevant information but minimize mutual information beyond that.
Subclasses SimCLRv1Module -- the backbone, projection head, and NT-Xent loss
are inherited unchanged. Only the augmentation policy differs.

Paper: "What Makes for Good Views for Contrastive Learning?"
Authors: Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, Phillip Isola
Venue: NeurIPS 2020
arXiv: https://arxiv.org/abs/2005.10243

Algorithm:
1. Use SimCLR backbone and NT-Xent loss unchanged.
2. Replace augmentation with aggressive "minimal MI" policy:
   - Color jitter s=1.5 (vs SimCLR's s=1.0)
   - Random grayscale p=0.4 (vs SimCLR's p=0.2)
   - NO Gaussian blur (removed entirely)
3. The InfoMin principle: views should share task-relevant information
   but minimize mutual information beyond that. More aggressive
   augmentation removes spurious correlations (texture, color bias).

Gotchas:
- This implementation is the augmentation-policy interpretation of InfoMin,
  not the full semi-supervised view-learning framework (deferred to v2).
- The augmentation override must be applied at data setup time, not just
  at module construction. Use build_augmentation() to get the transform.
- InfoMin augmentation may need batch_size tuning compared to SimCLR
  because harder augmentations can slow convergence.

Reference implementation: https://github.com/HobbitLong/PyContrast
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2

from core.config import InfoMinConfig, TrainConfig
from core.data import IMAGENET_MEAN, IMAGENET_STD, MultiViewTransform
from methods.simclr.module import SimCLRv1Module


class InfoMinModule(SimCLRv1Module):
    """InfoMin (Tian et al., NeurIPS 2020).

    What Makes for Good Views for Contrastive Learning?

    InfoMin is presented here as an augmentation-policy demonstration on top of
    the SimCLR/MoCo v2 backbone. The principle: views should share minimal
    mutual information except for task-relevant content. The implementation
    substitutes aggressive color jitter + random grayscale (and removes Gaussian
    blur) into the standard contrastive pipeline to approximate the "minimal-MI"
    recipe without the full semi-supervised view-learning framework.

    Paper: "What Makes for Good Views for Contrastive Learning?"
    Authors: Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan,
             Cordelia Schmid, Phillip Isola
    Venue: NeurIPS 2020
    arXiv: https://arxiv.org/abs/2005.10243

    Algorithm:
    1. Wrap a SimCLR-style backbone with the "InfoMin" augmentation policy
       (aggressive color jitter, random grayscale p=0.5, no Gaussian blur).
    2. Encode two augmented views and project via 2-layer MLP head.
    3. Compute symmetric NT-Xent loss as in SimCLR.

    Gotchas:
    - This module is the augmentation-policy demonstration only; the full
      semi-supervised view-learning framework is v2 scope (see V2-06).
    - Without the aggressive color jitter, the InfoMin recipe collapses to
      ordinary SimCLR — verify augmentation strength via tools/compare_augmentations.py.
    - Performance is sensitive to dataset domain — the default policy is tuned
      for natural images; medical/industrial domains may need different views.

    Reference implementation: https://github.com/HobbitLong/PyContrast
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        # Read InfoMin-specific config (or defaults)
        infomin_cfg = cfg.infomin or InfoMinConfig()
        self._color_strength = infomin_cfg.color_strength
        self._grayscale_prob = infomin_cfg.grayscale_prob
        self._use_blur = infomin_cfg.use_blur

    @classmethod
    def build_augmentation(cls, size: int = 224, color_strength: float = 1.5,
                           grayscale_prob: float = 0.4, use_blur: bool = False):
        """Build InfoMin-style aggressive augmentation (no Gaussian blur by default).

        Args:
            size: Crop size.
            color_strength: Color jitter multiplier (1.5 = more aggressive than SimCLR's 1.0).
            grayscale_prob: Random grayscale probability (0.4 vs SimCLR's 0.2).
            use_blur: Whether to include Gaussian blur (False for InfoMin).

        Returns:
            Callable augmentation transform.
        """
        s = color_strength
        transforms_list = [
            v2.RandomResizedCrop(size, scale=(0.2, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
            v2.RandomGrayscale(p=grayscale_prob),
        ]
        if use_blur:
            transforms_list.append(
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5)
            )
        transforms_list.extend([
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return v2.Compose(transforms_list)

    def setup(self, stage=None):
        """Set up dataset with InfoMin augmentation wired into the pipeline.

        Constructs the training dataset using InfoMinModule.build_augmentation()
        so that `method: infomin` in YAML automatically produces InfoMin-augmented
        training data. This is the production path per D-11 and D-12.
        """
        import os
        from torchvision.datasets import ImageFolder

        size = getattr(self.cfg, "size", 224)
        augmentation = self.build_augmentation(
            size=size,
            color_strength=self._color_strength,
            grayscale_prob=self._grayscale_prob,
            use_blur=self._use_blur,
        )
        transform = MultiViewTransform(augmentation, n_views=self.cfg.n_views)

        train_dir = os.path.join(self.cfg.data_dir, "train")
        if not os.path.isdir(train_dir):
            train_dir = self.cfg.data_dir
        self.train_dataset = ImageFolder(train_dir, transform=transform)
