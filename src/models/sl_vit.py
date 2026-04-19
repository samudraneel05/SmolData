# src/models/sl_vit.py
"""
SL-ViT: Shifted Patch Tokenization (SPT) + Locality Self-Attention (LSA)
Lee et al. — baseline model for Phase 1.

SPT shifts the image in 4 diagonal directions and concatenates channels
before standard patch projection — this injects local neighborhood info.
LSA adds a learnable local mask to attention weights to bias towards nearby tokens.
"""

import torch
import torch.nn as nn
from einops import rearrange


class ShiftedPatchTokenization(nn.Module):
    """
    SPT: Shift image in 4 diagonal directions, concat along channel dim,
    then project with a standard patch-conv.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        # After patchify: each token has 5 * in_channels * patch_size^2 features
        flat_dim = in_channels * 5 * patch_size * patch_size
        self.proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.num_patches = num_patches

    def _shift(self, x: torch.Tensor, shift: int, direction: str) -> torch.Tensor:
        """Roll image in a direction and zero-pad the rolled-out part."""
        if direction == "up":
            return torch.roll(x, -shift, dims=2)
        elif direction == "down":
            return torch.roll(x, shift, dims=2)
        elif direction == "left":
            return torch.roll(x, -shift, dims=3)
        elif direction == "right":
            return torch.roll(x, shift, dims=3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size // 2

        shifts = [
            self._shift(x, p, "up"),
            self._shift(x, p, "down"),
            self._shift(x, p, "left"),
            self._shift(x, p, "right"),
        ]
        x = torch.cat([x] + shifts, dim=1)  # (B, 5C, H, W)

        # Patchify: (B, 5C, H, W) → (B, N, 5C*p*p)
        x = rearrange(
            x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=self.patch_size, p2=self.patch_size
        )
        # Project to embed_dim
        return self.proj(x)


class LocalitySelfAttention(nn.Module):
    """
    LSA: standard MHSA with learnable diagonal attention scaling.
    Diagonal of attention matrix gets a learned temperature,
    enforcing locality bias without hard constraints.
    """

    def __init__(
        self, dim: int, num_heads: int = 8, attn_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        # Learnable temperature (locality bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply learned temperature to diagonal (self-attention entries)
        diag_mask = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(0)
        attn = attn - self.temperature * diag_mask * attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SLViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalitySelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SLViT(nn.Module):
    """SL-ViT: SPT + LSA — Lee et al. baseline for Phase 1."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 9,
        num_heads: int = 12,
        mlp_ratio: float = 2.0,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = ShiftedPatchTokenization(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.Sequential(*[SLViTBlock(embed_dim, num_heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))
