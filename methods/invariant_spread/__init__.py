"""Invariant Spread method package — Ye et al., CVPR 2019.

Registers 'invariant_spread' with the method dispatcher on import.
"""
from core.dispatcher import register_method
from methods.invariant_spread.module import InvariantSpreadModule

register_method("invariant_spread", InvariantSpreadModule)
