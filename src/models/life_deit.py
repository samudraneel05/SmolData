# src/models/life_deit.py
"""
DeiT-Tiny / DeiT-Small with LIFE Q/K/V projections.

This is Variant C/D's architecture for the DeiT backbone.
We patch timm's DeiT to replace the standard nn.Linear Q/K/V
with LIFEProjection at every transformer block.
"""

from typing import Sequence

import torch
import torch.nn as nn
from .life_module import LIFEAttention
from .vit_scratch import vit_tiny, vit_small


def _patch_model_with_life(
    model: nn.Module,
    kernel_sizes: Sequence[int] = (1, 3, 5),
    img_size: int = 32,
    patch_size: int = 4,
) -> nn.Module:
    """
    Replace every Attention block in a model with LIFEAttention.
    Returns the patched model in-place.
    """
    H = W = img_size // patch_size  # spatial patch grid dimensions

    for name, module in model.named_modules():
        if type(module).__name__ == "Attention":
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)

            dim = module.num_heads * (module.head_dim if hasattr(module, "head_dim") else module.scale ** -2)
            # Infer dim from qkv weight if available
            if hasattr(module, "qkv"):
                dim = module.qkv.in_features
            num_heads = module.num_heads

            life_attn = LIFEAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=0.0,
                proj_drop=0.0,
                kernel_sizes=kernel_sizes,
            )
            setattr(parent, child_name, life_attn)

    return model


class LIFEViT(nn.Module):
    """
    ViT with LIFE: patches all attention blocks in our base ViTScratch
    to use LIFE multi-scale Q/K/V projection.
    """

    def __init__(
        self,
        base_model: nn.Module,
        img_size: int = 32,
        patch_size: int = 4,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.backbone = base_model
        _patch_model_with_life(self.backbone, kernel_sizes, img_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def life_deit_tiny(num_classes: int = 10, img_size: int = 32) -> LIFEViT:
    base = vit_tiny(num_classes=num_classes, img_size=img_size)
    return LIFEViT(base, img_size, patch_size=4)


def life_deit_small(num_classes: int = 10, img_size: int = 32) -> LIFEViT:
    base = vit_small(num_classes=num_classes, img_size=img_size)
    return LIFEViT(base, img_size, patch_size=4)
