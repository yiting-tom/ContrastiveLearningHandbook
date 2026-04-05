"""MoCo v1 and v2 method registration.

Importing this package triggers dispatcher registration so that
``method_dispatcher(cfg)`` can instantiate MoCoV1Module or MoCoV2Module
from YAML config with ``method: moco_v1`` or ``method: moco_v2``.
"""
from core.dispatcher import register_method
from methods.moco.module import MoCoV1Module, MoCoV2Module

register_method("moco_v1", MoCoV1Module)
register_method("moco_v2", MoCoV2Module)
