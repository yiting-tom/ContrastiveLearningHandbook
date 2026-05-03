# core package — public API re-exports
# Each import is wrapped in try/except so that parallel execution of
# individual plans does not cause ImportError if a sibling module is not
# yet created.

try:
    from core.base import BaseSSLModule
except ImportError:
    pass

try:
    from core.backbone import build_backbone
except ImportError:
    pass

try:
    from core.config import TrainConfig, EvalConfig, load_config
except ImportError:
    pass

try:
    from core.data import (
        ContrastiveAugmentation,
        MultiCropDataset,
        MultiViewTransform,
        SSLDataModule,
    )
except ImportError:
    pass

try:
    from core.dispatcher import method_dispatcher
except ImportError:
    pass

try:
    from core.ema import EMAUpdater
except ImportError:
    pass

try:
    from core.losses import InfoNCELoss, SupConLoss
except ImportError:
    pass

try:
    from core.memory_bank import MemoryBank
except ImportError:
    pass

try:
    from core.queue import MomentumQueue
except ImportError:
    pass

try:
    from core.optimizers import LARS
except ImportError:
    pass

try:
    from core.projection import PredictorHead, ProjectionHead
except ImportError:
    pass

__all__ = [
    "BaseSSLModule",
    "build_backbone",
    "TrainConfig",
    "EvalConfig",
    "load_config",
    "ContrastiveAugmentation",
    "MultiCropDataset",
    "MultiViewTransform",
    "SSLDataModule",
    "method_dispatcher",
    "EMAUpdater",
    "InfoNCELoss",
    "SupConLoss",
    "LARS",
    "MemoryBank",
    "MomentumQueue",
    "PredictorHead",
    "ProjectionHead",
]
