"""Method dispatcher for SSL method instantiation.

This module provides a global registry that maps method name strings (as used
in YAML config files) to ``BaseSSLModule`` subclasses.  It is the single entry
point for all method instantiation.

Usage::

    from core.dispatcher import method_dispatcher, register_method

    # In a method module (e.g., methods/simclr_v1.py):
    register_method("simclr_v1", SimCLRv1)

    # In the training script:
    model = method_dispatcher(cfg)  # returns SimCLRv1(cfg)

Phases 2–8 each call ``register_method()`` in their ``__init__.py`` to add
their implementations.  The dispatcher itself never needs to change.
"""
from __future__ import annotations

from core.config import TrainConfig
from core.base import BaseSSLModule


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

# Maps method name strings to BaseSSLModule subclasses.
# Phases 2–8 will import and call register_method() to add their implementations.
_METHOD_REGISTRY: dict[str, type[BaseSSLModule]] = {}


def register_method(name: str, cls: type[BaseSSLModule]) -> None:
    """Register a method class in the dispatcher.

    Called by each method module (typically in its ``__init__.py``) to make
    the method available for dispatch.

    Args:
        name: Method name as used in config YAML (e.g., ``'simclr_v1'``,
            ``'moco_v1'``).
        cls: ``BaseSSLModule`` subclass to instantiate for this method.

    Raises:
        ValueError: If ``name`` is already registered.  Prevents accidental
            overwrites that could silently produce wrong behaviour.
    """
    if name in _METHOD_REGISTRY:
        raise ValueError(
            f"Method '{name}' is already registered. "
            f"Each method name must be unique across the registry."
        )
    _METHOD_REGISTRY[name] = cls


def method_dispatcher(cfg: TrainConfig) -> BaseSSLModule:
    """Instantiate the correct SSL method module from config.

    Looks up ``cfg.method`` in the registry and instantiates the corresponding
    ``BaseSSLModule`` subclass with ``cfg`` as its sole constructor argument.

    Args:
        cfg: Fully-validated training configuration.  ``cfg.method`` determines
            which registered method class to instantiate.

    Returns:
        An instance of the ``BaseSSLModule`` subclass registered under
        ``cfg.method``.

    Raises:
        ValueError: If ``cfg.method`` is not in the registry.  The error
            message includes the sorted list of available method names so
            the user knows what to choose from.

    Example::

        register_method("simclr_v1", SimCLRv1)
        cfg = TrainConfig(method="simclr_v1", ...)
        model = method_dispatcher(cfg)  # returns SimCLRv1(cfg)
    """
    if cfg.method not in _METHOD_REGISTRY:
        available = sorted(_METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown method: '{cfg.method}'. "
            f"Available methods: {available}"
        )
    return _METHOD_REGISTRY[cfg.method](cfg)


def available_methods() -> list[str]:
    """Return sorted list of registered method names.

    Returns:
        Alphabetically-sorted list of all method names currently in the
        registry.  Useful for building help text or validation error messages.
    """
    return sorted(_METHOD_REGISTRY.keys())
