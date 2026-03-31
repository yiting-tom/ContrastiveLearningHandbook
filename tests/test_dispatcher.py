"""Tests for core/dispatcher.py — method_dispatcher factory.

Tests verify:
  - ValueError is raised for unknown methods
  - Error message contains "Unknown method"
  - Error message lists available methods
  - Registering and dispatching a new method works (extensibility)
  - Returned instance is a BaseSSLModule subclass
  - Duplicate registration raises ValueError
  - available_methods() returns sorted list
"""
from __future__ import annotations

import pytest

from core.config import TrainConfig
from core.base import BaseSSLModule
from core.dispatcher import method_dispatcher, register_method, available_methods


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_registry():
    """Restore _METHOD_REGISTRY to its original state after each test.

    This prevents test pollution — a method registered in one test must not
    appear in the next test's registry.
    """
    from core.dispatcher import _METHOD_REGISTRY
    original = _METHOD_REGISTRY.copy()
    yield
    _METHOD_REGISTRY.clear()
    _METHOD_REGISTRY.update(original)


def _make_cfg(method: str = "test_method") -> TrainConfig:
    """Helper: create a minimal valid TrainConfig with the given method."""
    return TrainConfig(method=method)


# ---------------------------------------------------------------------------
# Dummy concrete subclass for testing
# ---------------------------------------------------------------------------

class DummyMethod(BaseSSLModule):
    """Minimal concrete subclass of BaseSSLModule for dispatcher tests."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)

    def build_projector(self):
        import torch.nn as nn
        return nn.Identity()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unknown_method_raises():
    """method_dispatcher raises ValueError for an unregistered method."""
    cfg = _make_cfg(method="nonexistent_method")
    with pytest.raises(ValueError):
        method_dispatcher(cfg)


def test_unknown_method_error_message_contains_unknown_method():
    """ValueError message contains 'Unknown method'."""
    cfg = _make_cfg(method="nonexistent_method")
    with pytest.raises(ValueError, match="Unknown method"):
        method_dispatcher(cfg)


def test_error_lists_available_methods():
    """ValueError message lists the names of registered methods."""
    register_method("dummy_alpha", DummyMethod)
    cfg = _make_cfg(method="does_not_exist")
    with pytest.raises(ValueError, match="dummy_alpha"):
        method_dispatcher(cfg)


def test_register_and_dispatch():
    """Registering a new method and dispatching to it returns the correct instance."""
    register_method("test_dummy", DummyMethod)
    cfg = _make_cfg(method="test_dummy")
    result = method_dispatcher(cfg)
    assert isinstance(result, DummyMethod)


def test_dispatch_returns_base_ssl_module():
    """method_dispatcher returns an instance of BaseSSLModule."""
    register_method("test_base_check", DummyMethod)
    cfg = _make_cfg(method="test_base_check")
    result = method_dispatcher(cfg)
    assert isinstance(result, BaseSSLModule)


def test_duplicate_registration_raises():
    """Registering the same method name twice raises ValueError."""
    register_method("test_dup", DummyMethod)
    with pytest.raises(ValueError):
        register_method("test_dup", DummyMethod)


def test_available_methods_returns_sorted_list():
    """available_methods() returns a sorted list of registered method names."""
    register_method("zebra_method", DummyMethod)
    register_method("apple_method", DummyMethod)
    register_method("mango_method", DummyMethod)
    methods = available_methods()
    assert "zebra_method" in methods
    assert "apple_method" in methods
    assert "mango_method" in methods
    assert methods == sorted(methods), "available_methods() must return a sorted list"
