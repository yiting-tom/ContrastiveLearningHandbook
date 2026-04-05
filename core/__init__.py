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
    from core.data import ContrastiveAugmentation, MultiViewTransform, SSLDataModule
except ImportError:
    pass

try:
    from core.ema import EMAUpdater
except ImportError:
    pass

try:
    from core.losses import InfoNCELoss
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
    from core.projection import ProjectionHead
except ImportError:
    pass

__all__ = [
    "BaseSSLModule",
    "build_backbone",
    "TrainConfig",
    "EvalConfig",
    "load_config",
    "ContrastiveAugmentation",
    "MultiViewTransform",
    "SSLDataModule",
    "EMAUpdater",
    "InfoNCELoss",
    "LARS",
    "MemoryBank",
    "MomentumQueue",
    "ProjectionHead",
]
