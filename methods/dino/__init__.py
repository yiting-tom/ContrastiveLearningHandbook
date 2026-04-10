"""DINO method package.

Importing this module triggers register_method("dino", DINOModule) via the
dispatcher registry, making "dino" available as a selectable method in TrainConfig.
"""
from core.dispatcher import register_method
from methods.dino.module import DINOModule

register_method("dino", DINOModule)
