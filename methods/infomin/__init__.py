"""InfoMin method package -- Tian et al., NeurIPS 2020.

Registers 'infomin' with the method dispatcher on import.
"""
from core.dispatcher import register_method
from methods.infomin.module import InfoMinModule

register_method("infomin", InfoMinModule)
