"""Instance Discrimination method package (Wu et al., CVPR 2018).

Registers 'instance_discrimination' with the method dispatcher on import.
"""
from core.dispatcher import register_method
from methods.instance_discrimination.module import InstanceDiscriminationModule

register_method("instance_discrimination", InstanceDiscriminationModule)
