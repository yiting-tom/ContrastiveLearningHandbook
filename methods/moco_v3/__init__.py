"""MoCo v3 method package.

Registers MoCoV3Module with the method dispatcher so it can be selected
via ``method: moco_v3`` in YAML configs.
"""
from core.dispatcher import register_method
from methods.moco_v3.module import MoCoV3Module

register_method("moco_v3", MoCoV3Module)
