"""Barlow Twins method registration."""
from core.dispatcher import register_method
from methods.barlow_twins.module import BarlowTwinsModule

register_method("barlow_twins", BarlowTwinsModule)
