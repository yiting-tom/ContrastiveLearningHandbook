"""SimCLR method package — Chen et al., ICML 2020 / NeurIPS 2020.

Registers 'simclr_v1' and 'simclr_v2' with the method dispatcher on import.
"""
from core.dispatcher import register_method
from methods.simclr.module import SimCLRv1Module, SimCLRv2Module

register_method("simclr_v1", SimCLRv1Module)
register_method("simclr_v2", SimCLRv2Module)
