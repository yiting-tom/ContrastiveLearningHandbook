"""Class Activation Map (CAM) visualization for SSL checkpoints.

Uses EigenCAM by default (gradient-free, works without a classifier --
correct default for SSL models). Switches to GradCAM when a downstream
classifier is present.

Architecture-aware target layer selection:
  - ResNet: backbone.layer4[-1]
  - ViT: backbone.blocks[-1].norm1 (with reshape_transform)

Usage:
    python eval/cam_vis.py configs/simclr_v1_resnet18.yaml --ckpt outputs/run/checkpoints/epoch-99.ckpt
    python eval/cam_vis.py configs/simclr_v1_resnet18.yaml --ckpt outputs/run/checkpoints/epoch-99.ckpt --classifier outputs/finetune/model.ckpt
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from core.config import TrainConfig, CAMConfig
from core.dispatcher import get_method


# ---------------------------------------------------------------------------
# Architecture-aware target layer selection
# ---------------------------------------------------------------------------

def get_target_layer(backbone: nn.Module, backbone_name: str) -> list[nn.Module]:
    """Select the appropriate CAM target layer based on backbone architecture.

    Args:
        backbone: The backbone module (already loaded and ready).
        backbone_name: Architecture name string (e.g., 'resnet18', 'vit_small_patch16_224').

    Returns:
        List containing the target layer for CAM computation.

    Raises:
        ValueError: If backbone_name is not a supported architecture.
    """
    name_lower = backbone_name.lower()
    if "resnet" in name_lower:
        return [backbone.layer4[-1]]
    elif "vit" in name_lower:
        return [backbone.blocks[-1].norm1]
    else:
        raise ValueError(
            f"Unsupported backbone for CAM: {backbone_name}. "
            f"Supported: resnet*, vit*"
        )


# ---------------------------------------------------------------------------
# ViT reshape transform
# ---------------------------------------------------------------------------

def vit_reshape_transform(
    tensor: torch.Tensor,
    height: int = 14,
    width: int = 14,
) -> torch.Tensor:
    """Reshape ViT [B, N+1, D] activations to [B, D, H, W] spatial maps.

    ViT produces sequence output at target layers. The first token is the
    CLS token which is discarded; the remaining N=H*W patch tokens are
    reshaped into a 2D spatial grid for CAM computation.

    Args:
        tensor: ViT layer output of shape [B, N_patches+1, D].
        height: Spatial height of the patch grid (default 14 for 224/16 input).
        width: Spatial width of the patch grid (default 14 for 224/16 input).

    Returns:
        Tensor of shape [B, D, H, W] suitable for CAM.
    """
    # Remove CLS token (first position)
    result = tensor[:, 1:, :]
    # Reshape from [B, H*W, D] to [B, H, W, D]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    # Permute to BCHW: [B, H, W, D] -> [B, D, H, W]
    result = result.permute(0, 3, 1, 2)
    return result


# ---------------------------------------------------------------------------
# CAM method selection
# ---------------------------------------------------------------------------

_CAM_METHODS = {
    "eigencam": EigenCAM,
    "gradcam": GradCAM,
}


def get_cam_method(method_name: str):
    """Return the CAM class for the given method name.

    Args:
        method_name: Method name string (e.g., 'eigencam', 'gradcam').

    Returns:
        CAM class (EigenCAM or GradCAM).

    Raises:
        ValueError: If method_name is not in the supported set.
    """
    if method_name not in _CAM_METHODS:
        raise ValueError(
            f"Unknown CAM method: {method_name!r}. "
            f"Supported methods: {sorted(_CAM_METHODS.keys())}"
        )
    return _CAM_METHODS[method_name]


# ---------------------------------------------------------------------------
# WrapperModule for GradCAM + classifier
# ---------------------------------------------------------------------------

class WrapperModule(nn.Module):
    """Wraps backbone + classifier head into a single module returning logits.

    Used when GradCAM requires gradient flow through a classifier to produce
    meaningful activation maps. The wrapper makes the combined model look like
    a standard classifier to the pytorch-grad-cam API.

    Args:
        backbone: SSL backbone module.
        head: Linear classification head (nn.Linear).
    """

    def __init__(self, backbone: nn.Module, head: nn.Linear) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------

def load_classifier(
    classifier_path: str,
    feat_dim: int,
    num_classes: int,
    device: str,
) -> nn.Linear:
    """Load a classifier head from a checkpoint file.

    Supports Lightning-format checkpoints (with 'state_dict' key) and
    plain PyTorch checkpoints containing a linear head.

    Args:
        classifier_path: Path to the classifier checkpoint file.
        feat_dim: Input feature dimension for the linear head.
        num_classes: Number of output classes.
        device: Device string for loading.

    Returns:
        Initialized and loaded nn.Linear classifier on the given device.
    """
    ckpt = torch.load(classifier_path, map_location=device, weights_only=True)

    # Extract state dict (Lightning format uses 'state_dict' key)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Filter to linear/head keys and strip prefix
    head_keys = {k for k in state_dict if "linear" in k or "head" in k}
    if not head_keys:
        raise ValueError(
            f"No linear/head keys found in classifier checkpoint. "
            f"Available keys: {sorted(state_dict.keys())}"
        )

    # Build linear head and load weights
    head = nn.Linear(feat_dim, num_classes)
    # Strip prefix, keep only weight and bias
    stripped = {}
    for k, v in state_dict.items():
        if "linear" in k or "head" in k:
            # e.g. "linear.weight" -> "weight", "head.weight" -> "weight"
            param_name = k.split(".")[-1]
            if param_name in ("weight", "bias"):
                stripped[param_name] = v
    head.load_state_dict(stripped)
    return head.to(device).eval()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _to_rgb_float(image: np.ndarray) -> np.ndarray:
    """Convert uint8 numpy array [H, W, 3] to float [H, W, 3] in [0, 1]."""
    img = image.astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main CAM computation
# ---------------------------------------------------------------------------

def run_cam(
    model: nn.Module,
    backbone_name: str,
    images: list[np.ndarray],
    cam_cfg: CAMConfig,
    output_dir: Path,
    classifier: nn.Linear | None = None,
) -> list[Path]:
    """Compute and save CAM overlay images.

    Args:
        model: Model with a .backbone attribute (SSL module or wrapper).
        backbone_name: Architecture name for target layer selection.
        images: List of uint8 numpy arrays [H, W, 3].
        cam_cfg: CAM configuration (method, n_images).
        output_dir: Directory to save PNG overlays.
        classifier: Optional loaded linear classifier for GradCAM path.

    Returns:
        List of saved PNG file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cap images to n_images limit (T-09-11: prevent OOM from large batches)
    images = images[:cam_cfg.n_images]

    # Determine reshape transform (ViT only)
    name_lower = backbone_name.lower()
    reshape_transform = vit_reshape_transform if "vit" in name_lower else None

    # Select CAM class
    CamClass = get_cam_method(cam_cfg.method)

    # Build CAM instance based on method and whether classifier is provided
    if cam_cfg.method == "gradcam" and classifier is not None:
        # GradCAM with classifier: wrap backbone + head
        wrapper = WrapperModule(model.backbone, classifier)
        target_layers = get_target_layer(wrapper.backbone, backbone_name)
        cam = CamClass(
            model=wrapper,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )
    elif cam_cfg.method == "gradcam" and classifier is None:
        # GradCAM without classifier: fall back to EigenCAM with warning
        warnings.warn(
            "GradCAM requires a classifier for meaningful maps on SSL models. "
            "Falling back to EigenCAM (gradient-free). "
            "Use --classifier to enable GradCAM.",
            UserWarning,
            stacklevel=2,
        )
        target_layers = get_target_layer(model.backbone, backbone_name)
        cam = EigenCAM(
            model=model.backbone,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )
    else:
        # Default: EigenCAM (gradient-free, works without classifier)
        target_layers = get_target_layer(model.backbone, backbone_name)
        cam = CamClass(
            model=model.backbone,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    saved_paths: list[Path] = []

    for i, img_np in enumerate(images):
        # Convert to PIL for resize, then to tensor
        pil_img = Image.fromarray(img_np)
        pil_img = pil_img.resize((224, 224))
        img_np_resized = np.array(pil_img)

        # Preprocess for model input
        input_tensor = _preprocess(pil_img).unsqueeze(0)

        # Float [0, 1] for overlay
        rgb_float = _to_rgb_float(img_np_resized)

        # Compute CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        cam_overlay = show_cam_on_image(rgb_float, grayscale_cam[0], use_rgb=True)

        # Save overlay
        out_path = output_dir / f"cam_{i:02d}.png"
        plt.imsave(str(out_path), cam_overlay)
        saved_paths.append(out_path)

    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAM visualizations for an SSL checkpoint."
    )
    parser.add_argument("config", type=str, help="Path to YAML training config")
    parser.add_argument("--ckpt", type=str, required=True, help="SSL checkpoint path")
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help=(
            "Path to a finetuned/linear-probe checkpoint containing the "
            "classification head (enables GradCAM mode)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CAM overlay PNGs (default: sibling of --ckpt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def main() -> None:
    """Load SSL checkpoint and generate CAM overlay images."""
    args = get_args()

    # Load config
    with open(args.config) as f:
        cfg = TrainConfig.model_validate(yaml.safe_load(f))

    # Import methods to populate the dispatcher registry
    import methods  # noqa: F401

    # Load SSL checkpoint
    MethodClass = get_method(cfg.method)
    model = MethodClass.load_from_checkpoint(args.ckpt, cfg=cfg)
    model.eval()
    model.to(args.device)

    # CAM config from YAML or defaults
    cam_cfg = (
        cfg.eval.cam
        if cfg.eval is not None and cfg.eval.cam is not None
        else CAMConfig()
    )

    # Output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.ckpt).parent.parent / "cam"
    )

    # Load classifier for GradCAM path (optional)
    classifier = None
    if args.classifier is not None:
        feat_dim = model.backbone.num_features
        # Determine num_classes from classifier checkpoint dimensions
        ckpt = torch.load(args.classifier, map_location=args.device, weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else {}
        # Find head weight shape
        num_classes = None
        for k, v in state_dict.items():
            if ("linear" in k or "head" in k) and k.endswith(".weight"):
                num_classes = v.shape[0]
                break
        if num_classes is None:
            raise ValueError(
                f"Could not determine num_classes from classifier checkpoint. "
                f"Keys: {sorted(state_dict.keys())}"
            )
        classifier = load_classifier(args.classifier, feat_dim, num_classes, args.device)

    # Load raw images from data_dir using PIL
    import os
    from torchvision.datasets.folder import IMG_EXTENSIONS
    data_dir = Path(cfg.data_dir)
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if any(fname.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                image_paths.append(os.path.join(root, fname))
        if len(image_paths) >= cam_cfg.n_images:
            break
    image_paths = image_paths[:cam_cfg.n_images]

    if not image_paths:
        raise RuntimeError(f"No images found in data_dir: {data_dir}")

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(np.array(img))

    # Run CAM
    saved_paths = run_cam(
        model=model,
        backbone_name=cfg.backbone,
        images=images,
        cam_cfg=cam_cfg,
        output_dir=output_dir,
        classifier=classifier,
    )

    for p in saved_paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
