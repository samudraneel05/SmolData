# src/models/__init__.py
"""Model registry for SmolData experiments."""

from .vit_scratch import ViTScratch, vit_tiny, vit_small, vit_base_paper1
from .resnet import SmallResNet, resnet18, resnet56
from .sl_vit import SLViT
from .life_module import LIFEProjection, LIFEAttention
from .life_deit import LIFEDeiT, life_deit_tiny, life_deit_small

__all__ = [
    "ViTScratch", "vit_tiny", "vit_small", "vit_base_paper1",
    "SmallResNet", "resnet18", "resnet56",
    "SLViT",
    "LIFEProjection", "LIFEAttention",
    "LIFEDeiT", "life_deit_tiny", "life_deit_small",
]
