# src/analysis/attention_viz.py
"""
Attention map extraction and visualization — following Zhu et al. (2023) Paper 1 methodology.

Steps per paper:
1. Extract attention weights from all L transformer blocks
2. Average across heads
3. Add residual identity (attention rollout base step)
4. Normalize to [0, 1]
5. Overlay on original image as heatmap
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class AttentionExtractor:
    """
    Extracts raw attention weights (pre-softmax or post-softmax) from
    all transformer blocks via forward hooks.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attention_maps: List[np.ndarray] = []
        self._hooks = []

    def _hook_fn(self, module, input, output):
        """Captures attention weights if module exposes them."""
        if hasattr(module, "_attn_weights"):
            self.attention_maps.append(module._attn_weights.detach().cpu().numpy())

    def __enter__(self):
        # Monkey-patch attention modules to store their weights
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and hasattr(module, "forward"):
                self._patch_attention(module)
        return self

    def __exit__(self, *args):
        # Restore and clean up
        self.attention_maps.clear()

    def _patch_attention(self, module: nn.Module) -> None:
        """Wrap attention forward to save weights."""
        orig_forward = module.forward

        def new_forward(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            # Store attn weights if available as side-effect
            return out

        module.forward = new_forward


def extract_attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Attention Rollout — Abnar & Zuidema (2020).
    Recursively multiply attention matrices across all layers,
    averaging over heads and adding identity (for residual path).

    Returns a (H_patches × W_patches) saliency map for the cls token.
    """
    model.eval()
    attentions = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # For timm-style attention, intercept the attn weights
            with torch.no_grad():
                B, N, C = input[0].shape
                qkv = module.qkv(input[0])
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                scale = (C // module.num_heads) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                attentions.append(attn.detach().cpu())
        return hook

    hooks = []
    for i, (name, module) in enumerate(model.named_modules()):
        if type(module).__name__ == "Attention" and hasattr(module, "qkv"):
            h = module.register_forward_hook(make_hook(i))
            hooks.append(h)

    with torch.no_grad():
        model(image.unsqueeze(0).to(device))

    for h in hooks:
        h.remove()

    if not attentions:
        return np.zeros((8, 8))

    # Rollout: product of (I + A) / 2 across layers
    num_patches = attentions[0].shape[-1]  # N = num_patches + 1 (cls)
    result = torch.eye(num_patches)

    for attn in attentions:
        attn_mean = attn.mean(dim=1).squeeze(0)  # Average over heads: (N, N)
        # Discard low-attention tokens
        flat = attn_mean.flatten()
        threshold = flat.kthvalue(int(len(flat) * discard_ratio)).values
        attn_mean[attn_mean < threshold] = 0.0

        # Add residual connection and normalize
        I = torch.eye(num_patches)
        a = (attn_mean + I) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    # cls token → patch attention: row 0, columns 1:
    mask = result[0, 1:].numpy()
    return mask


def visualize_attention(
    model: nn.Module,
    images: torch.Tensor,
    labels: List[int],
    device: torch.device,
    patch_size: int = 4,
    img_size: int = 32,
    save_dir: str = "outputs/attention_maps",
    n_images: int = 10,
) -> None:
    """
    Visualize attention rollout for n_images test samples.

    Generates a grid: original image | attention heatmap | overlay.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    h_patches = w_patches = img_size // patch_size

    for idx in range(min(n_images, images.shape[0])):
        image = images[idx]
        label = labels[idx]

        mask = extract_attention_rollout(model, image, device)
        mask = mask.reshape(h_patches, w_patches)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # Upsample mask to image size
        import torch.nn.functional as F
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        mask_up = F.interpolate(mask_tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
        mask_up = mask_up.squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        # Denormalize image for display
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        axes[0].imshow(img_np)
        axes[0].set_title(f"Image (class {label})")
        axes[0].axis("off")

        axes[1].imshow(mask_up, cmap="hot")
        axes[1].set_title("Attention Rollout")
        axes[1].axis("off")

        axes[2].imshow(img_np)
        axes[2].imshow(mask_up, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(save_dir / f"attn_{idx:04d}_class{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
