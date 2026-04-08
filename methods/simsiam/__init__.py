"""SimSiam method registration."""
from core.dispatcher import register_method
from methods.simsiam.module import SimSiamModule

register_method("simsiam", SimSiamModule)
