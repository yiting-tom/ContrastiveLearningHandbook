"""SupCon method package — Khosla et al., NeurIPS 2020.

Registers 'supcon' (stage-1 pretraining) and 'supcon_finetune' (stage-2
linear fine-tuning) with the method dispatcher on import.

Two-stage workflow:
  Stage 1: python train.py --config configs/supcon_stage1_resnet18.yaml
  Stage 2: python train.py --config configs/supcon_stage2_resnet18.yaml
"""
from core.dispatcher import register_method
from methods.supcon.module import SupConFinetuneModule, SupConModule

register_method("supcon", SupConModule)
register_method("supcon_finetune", SupConFinetuneModule)
