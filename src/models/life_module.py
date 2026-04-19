# src/models/life_module.py
"""
LIFE (Local Image Features Extractor) module — Akkaya et al. 2024
(Paper: 1-s2.0-S0031320324002619)

Replaces the standard nn.Linear Q/K/V projection in ViT attention blocks
with multi-scale depthwise-separable convolutions that explicitly encode
local spatial structure at three scales simultaneously.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise → pointwise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class LIFEProjection(nn.Module):
    """
    Multi-scale LIFE projection to replace nn.Linear in Q/K/V attention.

    Takes sequence tokens (B, N, C), reshapes to spatial (B, C, H, W),
    applies three parallel depthwise-sep conv branches at scales
    kernel_sizes = (1, 3, 5), concatenates, and projects back.

    Output shape: (B, N, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        assert out_dim % len(kernel_sizes) == 0, (
            f"out_dim ({out_dim}) must be divisible by number of branches ({len(kernel_sizes)})"
        )
        branch_dim = out_dim // len(kernel_sizes)
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv(in_dim, branch_dim, k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        # Final linear projection after concatenation
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x:  (B, N, C) — patch token sequence (N = H*W + optional cls token)
            H:  spatial height of patch grid
            W:  spatial width  of patch grid
        Returns:
            (B, N, out_dim)
        """
        B, N, C = x.shape
        has_cls = N == H * W + 1
        if has_cls:
            cls_token, tokens = x[:, :1, :], x[:, 1:, :]
        else:
            tokens = x

        # Reshape to (B, C, H, W) for conv
        spatial = rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)

        # Apply each conv branch and concatenate along channel dimension
        outs = [branch(spatial) for branch in self.branches]
        out = torch.cat(outs, dim=1)  # (B, out_dim, H, W)

        # Flatten back to sequence
        out = rearrange(out, "b c h w -> b (h w) c")

        if has_cls:
            # Project cls token with first branch's channel slice (dim-preserving linear)
            cls_out = self.proj(torch.cat([
                cls_token,
                torch.zeros(B, 1, out.shape[-1] - C, device=x.device)
            ], dim=-1)[:, :, :out.shape[-1]])
            out = torch.cat([cls_out, out], dim=1)

        return self.proj(out)


class LIFEAttention(nn.Module):
    """
    Multi-head self-attention with LIFE Q/K/V projections.
    Drop-in replacement for timm's Attention block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Three separate LIFE modules for Q, K, V
        self.q_life = LIFEProjection(dim, dim, kernel_sizes)
        self.k_life = LIFEProjection(dim, dim, kernel_sizes)
        self.v_life = LIFEProjection(dim, dim, kernel_sizes)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = self.v_bias = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_life(x, H, W)
        k = self.k_life(x, H, W)
        v = self.v_life(x, H, W)

        if self.q_bias is not None:
            q = q + self.q_bias
            v = v + self.v_bias

        def to_heads(t):
            return rearrange(t, "b n (h d) -> b h n d", h=self.num_heads)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(out))
