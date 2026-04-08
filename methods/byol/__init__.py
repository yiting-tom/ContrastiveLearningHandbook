"""BYOL method registration.

Importing this package triggers dispatcher registration so that
``method_dispatcher(cfg)`` can instantiate BYOLModule from YAML config
with ``method: byol``.
"""
from core.dispatcher import register_method
from methods.byol.module import BYOLModule

register_method("byol", BYOLModule)
