"""Tests for core/config.py — TrainConfig, EvalConfig, sub-configs, and load_config."""
import os
import tempfile

import pytest
import yaml
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Test 1: TrainConfig.model_validate(valid_dict) succeeds
# ---------------------------------------------------------------------------
def test_valid_config(toy_config_dict):
    """TrainConfig validates a minimal valid dict and returns a typed config object."""
    from core.config import TrainConfig

    cfg = TrainConfig.model_validate(toy_config_dict)
    assert cfg.method == "simclr_v1"
    assert cfg.backbone == "resnet18"
    assert cfg.pretrained is False
    assert cfg.max_epochs == 10
    assert cfg.warmup_epochs == 1
    assert cfg.batch_size == 32
    assert isinstance(cfg.lr, float)
    assert cfg.optimizer == "adamw"


# ---------------------------------------------------------------------------
# Test 2: Unknown top-level YAML key raises ValidationError
# ---------------------------------------------------------------------------
def test_unknown_key_raises(toy_config_dict):
    """Unknown keys at the top level raise a Pydantic ValidationError (extra='forbid')."""
    from core.config import TrainConfig

    bad = dict(toy_config_dict)
    bad["unknown_field"] = "oops"
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# Test 3: Unknown key in nested sub-config raises ValidationError
# ---------------------------------------------------------------------------
def test_nested_unknown_key_raises(toy_config_dict):
    """extra='forbid' propagates to nested sub-configs (e.g., simclr dict)."""
    from core.config import TrainConfig

    bad = dict(toy_config_dict)
    bad["simclr"] = {"temperature": 0.5, "projection_dim": 128, "bad_field": 99}
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# Test 4: load_config("configs/example.yaml") returns a valid TrainConfig
# ---------------------------------------------------------------------------
def test_load_config_from_yaml():
    """load_config reads configs/example.yaml and returns a valid TrainConfig."""
    from core.config import TrainConfig, load_config

    # Resolve path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(project_root, "configs", "example.yaml")
    cfg = load_config(yaml_path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.method == "simclr_v1"


# ---------------------------------------------------------------------------
# Test 5: EvalConfig with all 6 sub-schemas validates correctly
# ---------------------------------------------------------------------------
def test_eval_config_all_sub_schemas():
    """EvalConfig accepts all 6 eval tool sub-configs when provided."""
    from core.config import EvalConfig

    data = {
        "linear_probe": {"max_epochs": 50, "lr": 0.05, "milestones": [30, 40]},
        "knn": {"k": 100, "temperature": 0.1, "every_n_epochs": 10},
        "tsne": {"n_samples": 1000, "perplexities": [20, 40]},
        "umap": {"n_samples": 2000, "metric": "euclidean"},
        "finetune": {"backbone_lr": 5e-5, "head_lr": 5e-4, "freeze_bn": False},
        "cam": {"method": "eigencam", "n_images": 4},
    }
    cfg = EvalConfig.model_validate(data)
    assert cfg.linear_probe is not None
    assert cfg.knn is not None
    assert cfg.tsne is not None
    assert cfg.umap is not None
    assert cfg.finetune is not None
    assert cfg.cam is not None
    assert cfg.knn.k == 100


# ---------------------------------------------------------------------------
# Test 6: Per-method sub-configs are Optional and default to None
# ---------------------------------------------------------------------------
def test_method_sub_configs_optional(toy_config_dict):
    """All per-method sub-configs (simclr, moco, etc.) are Optional and default to None."""
    from core.config import TrainConfig

    cfg = TrainConfig.model_validate(toy_config_dict)
    assert cfg.simclr is None
    assert cfg.moco is None
    assert cfg.byol is None
    assert cfg.swav is None
    assert cfg.barlow_twins is None
    assert cfg.simsiam is None
    assert cfg.dino is None
    assert cfg.supcon is None


# ---------------------------------------------------------------------------
# Test 7: Invalid types raise ValidationError
# ---------------------------------------------------------------------------
def test_invalid_type_raises(toy_config_dict):
    """Invalid types (e.g., lr='not_a_float') raise a Pydantic ValidationError."""
    from core.config import TrainConfig

    bad = dict(toy_config_dict)
    bad["lr"] = "not_a_float"
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(bad)
