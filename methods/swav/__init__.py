"""SwAV method package -- Caron et al., NeurIPS 2020.

Registers 'swav' with the method dispatcher on import.
"""
from core.dispatcher import register_method
from methods.swav.module import SwAVModule

register_method("swav", SwAVModule)
