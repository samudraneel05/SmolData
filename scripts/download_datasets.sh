#!/bin/bash
# scripts/download_datasets.sh
# Auto-download CIFAR-10, CIFAR-100, SVHN
# Works on Kaggle, Colab, and local (no venv required)

set -e
mkdir -p data

python3 - <<'EOF'
import torchvision
print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10("./data", train=True, download=True)
torchvision.datasets.CIFAR10("./data", train=False, download=True)

print("Downloading CIFAR-100...")
torchvision.datasets.CIFAR100("./data", train=True, download=True)
torchvision.datasets.CIFAR100("./data", train=False, download=True)

print("Downloading SVHN...")
torchvision.datasets.SVHN("./data/svhn", split="train", download=True)
torchvision.datasets.SVHN("./data/svhn", split="test", download=True)

print("\nDone. For CINIC-10: https://datashare.ed.ac.uk/handle/10283/3192")
print("For Tiny-ImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
print("For CIFAR-10-C/100-C: https://zenodo.org/record/2535967")
EOF
