# src/models/vit_scratch.py
"""
Vanilla Vision Transformer (Dosovitskiy et al., 2021) trained from scratch.
Designed for SMALL datasets at NATIVE resolution (no 224×224 upscaling).

Key differences from the original ViT for small-data regime:
- Default patch_size=4 for 32×32 inputs (CIFAR), 8 for 64×64 (Tiny-ImageNet)
- 3×3 stem conv instead of non-overlapping patch embedding (optional)
- No class-imbalanced cls-token weighting
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Non-overlapping patch embedding. Optionally replace with SPT (see sl_vit.py)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)            # (B, D, h, w)
        return rearrange(x, "b d h w -> b (h w) d")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Standard multi-head self-attention (no LIFE modification)."""

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT model
# ---------------------------------------------------------------------------

class ViTScratch(nn.Module):
    """
    Vanilla ViT for small datasets.

    Default config mirrors Zhu et al. (2023) — Paper 1:
        9.6M parameters, depth=6, heads=8, embed_dim=512, patch_size=4 (CIFAR)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return self.norm(x[:, 0])   # cls token representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def vit_tiny(num_classes: int = 10, img_size: int = 32, **kwargs) -> ViTScratch:
    """ViT-Tiny from 'Vision Transformers for Small datasets':
    embed_dim=192, depth=9, heads=12, mlp_ratio=2.0 (approx 2.6M params)."""
    return ViTScratch(img_size=img_size, embed_dim=192, depth=9, num_heads=12,
                      mlp_ratio=2.0, num_classes=num_classes, **kwargs)


def vit_small(num_classes: int = 10, img_size: int = 32, **kwargs) -> ViTScratch:
    """ViT-Small: embed_dim=384, depth=12, heads=6."""
    return ViTScratch(img_size=img_size, embed_dim=384, depth=12, num_heads=6,
                      num_classes=num_classes, **kwargs)


def vit_base_paper1(num_classes: int = 10, img_size: int = 32, **kwargs) -> ViTScratch:
    """ViT config from Zhu et al. (2023) Paper 1: 9.6M params."""
    return ViTScratch(img_size=img_size, embed_dim=512, depth=6, num_heads=8,
                      num_classes=num_classes, **kwargs)
