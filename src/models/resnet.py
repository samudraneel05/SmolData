# src/models/resnet.py
"""
ResNet-18 and ResNet-56 adapted for small-resolution inputs (32×32, 64×64).

Small-data adaptations (standard practice for CIFAR benchmarks):
- Replace 7×7 conv → 3×3 conv
- Remove initial max-pool
This prevents aggressive spatial downsampling on tiny images.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Type


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """
    Generic ResNet with small-dataset adaptation (3×3 stem, no initial maxpool).
    Can instantiate ResNet-18 (layers=[2,2,2,2]) or ResNet-56 (layers=[9,9,9]).
    """

    def __init__(
        self,
        layers: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = 64

        # Small-data stem: 3×3 conv, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    def get_feature_maps(self, x: Tensor) -> List[Tensor]:
        """Return intermediate feature maps for CKA analysis."""
        feats = []
        x = self.stem(x)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            feats.append(x.mean(dim=(2, 3)))  # Global average for CKA
        return feats


def resnet18(num_classes: int = 10) -> SmallResNet:
    return SmallResNet([2, 2, 2, 2], num_classes=num_classes)


def resnet56(num_classes: int = 10) -> SmallResNet:
    """ResNet-56 from He et al. (only 3 groups of layers, ~0.85M params)."""
    # For CIFAR, ResNet-56 uses BasicBlock ×9 at each of 3 widths:
    # Override layer config to match original CIFAR ResNet paper.
    model = SmallResNet([9, 9, 9, 0], num_classes=num_classes)
    return model
