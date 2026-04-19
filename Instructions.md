# SmolData — Step-by-Step Execution Guide

> **Who this is for**: Running experiments across multiple Kaggle/Colab accounts.
> This tells you exactly what to do, cell by cell, and what to collect at the end.

---

## Big Picture First

You are running **four variants of a ViT model** on small datasets to see which combination of locality tricks works best. The four variants are:

| Label | What it does |
|-------|-------------|
| **A** | Plain ViT, trained normally from scratch — your control |
| **B** | Plain ViT, but first self-supervised pretrained on the same data (DINO) |
| **C** | ViT with LIFE (conv layers inside attention), trained normally |
| **D** | ViT with LIFE + self-supervised pretraining — **the proposed method** |

You run this on CIFAR-10 and CIFAR-100 (both auto-download). At the end you want to show **D > B, C > A**, with the gap being largest on CIFAR-100 (harder).

You have 3 phases to run across your accounts:

| Phase | What | Which account | Est. time |
|-------|------|--------------|-----------|
| **Phase 1** | Baseline models (ViT, ResNet, SL-ViT) | Account 1 | ~2h |
| **Phase 2** | The 4 variants on CIFAR-10 + CIFAR-100 | Account 2 | ~3h |
| **Phase 3** | OED complexity grid (just A vs D) | Account 3 | ~2h |
| **Phase 4** | CKA + attention pictures | Any account with Phase 2 checkpoints | ~30 min |

> **Kaggle limit**: 30 GPU-hours per account per week. Each session is max 12 hours.
> Each phase fits in one session. Use one account per phase to conserve quota.

---

## Before You Start: Push Your Code to GitHub

You've already committed. Make sure it's pushed:
```bash
git push origin main
```
You'll clone this repo inside every notebook.

---

## WandB Setup (Do This Once Before Any Experiment)

WandB (Weights & Biases) is an experiment tracker. Every training run will appear as a
row in a live dashboard at wandb.ai — you can see loss curves, accuracy, GPU usage,
and compare all 4 variants side-by-side without downloading any files.

**You only need to set this up once. It takes 5 minutes.**

### Step 1 — Create a WandB account

Go to **[wandb.ai](https://wandb.ai)** → Sign Up (use your GitHub account is easiest).

After signing in, go to **[wandb.ai/settings](https://wandb.ai/settings)** → scroll down to
**"Danger Zone"** → find **"API keys"** → click **"New key"** → copy the key.

It looks like: `abc1234def5678...` (40 characters)

> **Save this key somewhere safe.** You'll paste it into every Kaggle/Colab session.
> Do NOT paste it directly into your notebook code — use Secrets (explained below).

### Step 2 — Add WandB key as a Kaggle Secret

Do this once per Kaggle account. It permanently stores the key.

1. Go to kaggle.com → click your profile picture → **Settings**
2. Scroll down to **"API"** section → click **"Add a new secret"** (or the key icon)
3. **Name**: `WANDB_API_KEY`
4. **Value**: paste your 40-character API key
5. Click **Save**

Now in every notebook on that account, add this cell at the **very top**:

```python
# Cell 0 — Load WandB API key from Kaggle Secrets
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
print("WandB key loaded ✓")
```

> This must run before any `wandb` import. Put it as the first cell, before the
> `!pip install` cell.

### Step 2 (alternative) — Add WandB key in Google Colab

1. In Colab: left sidebar → click the **🔑 key icon** ("Secrets")
2. Click **"Add new secret"**
3. **Name**: `WANDB_API_KEY`
4. **Value**: your API key
5. Toggle **"Notebook access"** to ON

Then add this as the first cell:
```python
# Cell 0 — Load WandB API key from Colab Secrets
from google.colab import userdata
import os

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
print("WandB key loaded ✓")
```

### Step 3 — Turn on WandB in the config

Open `configs/lite.yaml` in your repo. Find line 39 and change it from `false` to `true`:

```yaml
# BEFORE (default — no tracking)
use_wandb: false

# AFTER — enables tracking for all runs
use_wandb: true
wandb_project: smoldata   # this is the project name on wandb.ai
```

Then commit and push this change:
```bash
git add configs/lite.yaml
git commit -m "enable wandb tracking"
git push origin main
```

> If you don't want to commit this, you can also just edit the file **inside the notebook**
> after cloning, before running experiments:
> ```python
> # Alternative: edit lite.yaml inside the notebook without committing
> import yaml
> with open("configs/lite.yaml") as f:
>     cfg = yaml.safe_load(f)
> cfg["use_wandb"] = True
> with open("configs/lite.yaml", "w") as f:
>     yaml.dump(cfg, f)
> print("WandB enabled in config ✓")
> ```

### Step 4 — Verify the connection works

Add this cell and run it before your training cells:
```python
import wandb, os
print("WANDB_API_KEY is set:", bool(os.environ.get("WANDB_API_KEY")))
wandb.login()
# Should print: "Successfully logged in to Weights & Biases!"

# Quick sanity check — lists your projects
api = wandb.Api()
print("Entity:", api.default_entity)
```

### What you'll see on wandb.ai

Once training starts, your runs will appear at:
**[wandb.ai/samudraneel05-motilal-nehru-national-institute-of-technology/smoldata](https://wandb.ai/samudraneel05-motilal-nehru-national-institute-of-technology/smoldata)**

Runs are named like `vA_vit_tiny_cifar10_s42`, `vD_vit_tiny_cifar100_s42`, etc.

Each run logs:
- **`train_loss`** and **`val_acc`** every epoch (live updating)
- **`best_val_acc`** at the end of the run
- All config values (variant, arch, dataset, fraction, LR, epochs, etc.)

**Most useful WandB views for your paper:**

1. **"Table" tab** → group by `variant` column → sort by `best_val_acc` → this is Table 2
2. **"Charts" tab** → plot `val_acc` grouped by `variant` → screenshot for paper Figure 1
3. **"Overview"** on any run → confirms all hyperparameters were identical across variants

### If WandB is blocked or you prefer no tracking

Leave `use_wandb: false` in `configs/lite.yaml`. All results still save to
`results/phase2_results.csv` automatically — WandB is optional.

---

## ACCOUNT 1 — Phase 1: Baseline Replication

**Goal**: Confirm your training setup works and roughly matches published numbers.

### Step 1 — Create a new Kaggle notebook

1. Go to kaggle.com → **Create** → **New Notebook**
2. Top right: **Session options** → GPU → **T4 GPU** → Save
3. Title it: `smoldata-phase1`

### Step 2 — Paste these cells in order

**Cell 0** — WandB API key (skip if not using WandB):
```python
# On Kaggle:
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
print("WandB key loaded ✓")

# On Colab instead use:
# from google.colab import userdata
# os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

**Cell 1** — Install and clone:
```python
# Install extra packages (torch/torchvision already on Kaggle)
!pip install timm einops hydra-core omegaconf wandb rich -q

# Clone your repo
!git clone https://github.com/YOUR_USERNAME/SmolData.git
%cd SmolData

# Editable install so relative imports work
!pip install -e . -q
```
> Replace `YOUR_USERNAME` with your actual GitHub username.

**Cell 2** — Download the datasets:
```python
!bash scripts/download_datasets.sh
```
This downloads CIFAR-10 (~170 MB) and CIFAR-100 (~170 MB) automatically.
Wait for it to finish before running anything else.

**Cell 3** — Run Phase 1 (3 models × 2 datasets = 6 training runs):
```python
import subprocess, sys

models = ["vit_scratch", "resnet18", "sl_vit"]
datasets = ["cifar10", "cifar100"]

for model in models:
    for dataset in datasets:
        print(f"\n{'='*50}\nRunning {model} on {dataset}\n{'='*50}")
        subprocess.run([
            sys.executable, "experiments/phase1_baselines.py",
            "--model", model,
            "--dataset", dataset,
            "--seed", "42",
            "--config", "configs/lite.yaml"
        ])
```
This takes ~90 minutes. Each model prints a PASS/FAIL against known targets.

**Cell 4** — Save results:
```python
import pandas as pd
df = pd.read_csv("results/phase1_baselines.csv")
print(df[["model", "dataset", "val_acc"]].to_string())
df.to_csv("/kaggle/working/phase1_results.csv", index=False)
print("\nSaved!")
```

### Step 3 — Download output

Right panel → **Output** → `phase1_results.csv` → Download.
Save locally as: `SmolData/results/phase1_baselines.csv`

---

## ACCOUNT 2 — Phase 2: The Four Variants (Main Experiment)

**Goal**: Run all 4 variants on both datasets. This is the core of your paper.

### Step 1 — Create notebook on Account 2, Title: `smoldata-phase2`

> **Remember**: Add the WandB secret to this Kaggle account too (same process as Account 1 — see "WandB Setup" section above). Run **Cell 0** first in every notebook.

### Step 2 — Cells

**Cell 1** — Setup:
```python
!pip install timm einops hydra-core omegaconf wandb rich -q
!git clone https://github.com/YOUR_USERNAME/SmolData.git
%cd SmolData
!pip install -e . -q
!bash scripts/download_datasets.sh
```

**Cell 2** — Run Phase 2 (8 total training runs):
```python
# Reads everything from configs/lite.yaml
# Settings: datasets=[cifar10, cifar100], archs=[vit_tiny], seeds=[42]
# 4 variants x 1 arch x 2 datasets x 1 seed = 8 runs

!python experiments/phase2_variants.py --config configs/lite.yaml
```

> **Timeline on T4**:
> - Variants A and C (no SSL): ~30-40 min each per dataset
> - Variants B and D (SSL pretraining first): ~50-60 min each per dataset
> - **Total: ~6-7 hours** (fits in one Kaggle session)
>
> Watch for logs like:
> `Running Variant A | vit_tiny | cifar10 (fraction=20%) | seed=42`

**Cell 3** — Download results + checkpoints:
```python
import pandas as pd, shutil

df = pd.read_csv("results/phase2_results.csv")
print(df[["variant", "dataset", "val_acc"]].to_string())
df.to_csv("/kaggle/working/phase2_results.csv", index=False)

# Zip checkpoints (needed for Phase 4)
shutil.make_archive("/kaggle/working/phase2_checkpoints", "zip", "checkpoints/phase2")
print("Done!")
```

### Step 3 — Download from Kaggle Output

1. `phase2_results.csv` → save to `SmolData/results/phase2_results.csv`
2. `phase2_checkpoints.zip` → unzip to `SmolData/checkpoints/phase2/`

---

## ACCOUNT 3 — Phase 3: OED Complexity Ablation

**Goal**: Find out which ViT hyperparameter matters most. You run 9 OED configs on
CIFAR-100 for Variants A and D only (= 18 runs).

### Step 1 — Create notebook on Account 3, Title: `smoldata-phase3`

### Step 2 — Cells

**Cell 1** — Setup:
```python
!pip install timm einops hydra-core omegaconf wandb rich -q
!git clone https://github.com/YOUR_USERNAME/SmolData.git
%cd SmolData
!pip install -e . -q
!bash scripts/download_datasets.sh
```

**Cell 2** — Print OED design table so you know what you're running:
```python
import sys
sys.path.insert(0, "src")
from src.analysis.complexity_oed import print_oed_table
print_oed_table()
```

**Cell 3** — Run all 18 OED experiments (takes ~2 hours):
```python
import sys, subprocess, yaml
sys.path.insert(0, "src")
from src.analysis.complexity_oed import get_oed_experiments

for exp in get_oed_experiments():
    for variant in ["A", "D"]:
        print(f"\n>>> OED Exp {exp.experiment_id} | Variant {variant} "
              f"| depth={exp.depth} heads={exp.num_heads} "
              f"patch={exp.patch_size} mlp={exp.mlp_ratio}")

        # Build a temporary config for this OED configuration
        temp_cfg = {
            "optimizer": "adam", "lr": 0.002, "weight_decay": 0.05,
            "epochs": 30, "warmup_epochs": 3, "batch_size": 64,
            "clip_grad": 1.0, "label_smoothing": 0.1,
            "mixup_alpha": 0.4, "cutmix_alpha": 1.0,
            "mixup_prob": 1.0, "mixup_switch_prob": 0.5,
            "ssl_epochs": 30, "ssl_lr": 0.0005, "ema_momentum": 0.996,
            "dino_out_dim": 8192, "n_crops": 4,
            "seeds": [42], "num_eval_samples_cka": 500,
            "num_workers": 2, "use_wandb": False, "use_amp": True,
            "dataset_fraction": 0.2,
            "datasets": ["cifar100"],
            "archs": ["vit_tiny"],
        }
        cfg_path = f"configs/oed_exp{exp.experiment_id}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(temp_cfg, f)

        subprocess.run([
            sys.executable, "experiments/phase2_variants.py",
            "--variant", variant,
            "--config", cfg_path,
        ])

print("\nOED complete!")
```

**Cell 4** — Save:
```python
import pandas as pd
df = pd.read_csv("results/phase2_results.csv")
# Phase 3 runs also append to phase2_results.csv
df.to_csv("/kaggle/working/oed_results.csv", index=False)
print(df[["variant", "dataset", "val_acc"]].to_string())
```

### Step 3 — Download `oed_results.csv`

---

## ACCOUNT 4 — Phase 4: CKA + Attention Maps

**Goal**: Understand mechanically WHY Variant D works. Show its lower layers
develop local (CNN-like) representations, unlike Variant A.

**You need the Phase 2 checkpoints first.** Upload them to Kaggle Datasets:

1. kaggle.com → **Datasets** → **New Dataset**
2. Title: `smoldata-phase2-ckpts`
3. Upload `phase2_checkpoints.zip` → **Create**

### Step 1 — Create notebook on any account with GPU quota

**Add data**: right panel → **Add data** → search `smoldata-phase2-ckpts` → Add

### Step 2 — Cells

**Cell 1** — Setup:
```python
!pip install timm einops hydra-core omegaconf wandb rich -q
!git clone https://github.com/YOUR_USERNAME/SmolData.git
%cd SmolData
!pip install -e . -q
!bash scripts/download_datasets.sh

# Copy checkpoints from attached dataset
import shutil, os
os.makedirs("checkpoints/phase2", exist_ok=True)
!cp -r /kaggle/input/smoldata-phase2-ckpts/* checkpoints/phase2/
!ls checkpoints/phase2/
```

**Cell 2** — Run CKA and attention visualization:
```python
!python experiments/phase4_analysis.py \
    --model_a checkpoints/phase2/vA_vit_tiny_cifar100_s42_best.pt \
    --model_d checkpoints/phase2/vD_vit_tiny_cifar100_s42_best.pt \
    --dataset cifar100 \
    --n_cka_samples 500 \
    --output_dir outputs/phase4
```

**Cell 3** — View figures inline:
```python
from IPython.display import Image, display
import os
for f in sorted(os.listdir("outputs/phase4")):
    if f.endswith(".png"):
        print(f"\n=== {f} ===")
        display(Image(f"outputs/phase4/{f}"))
```

**Cell 4** — Save figures:
```python
import shutil
shutil.make_archive("/kaggle/working/phase4_figures", "zip", "outputs/phase4")
```

### Step 3 — Download `phase4_figures.zip`. Extract to `SmolData/outputs/phase4/`

---

## Building Results Tables Locally

After downloading all CSVs, run these in a local notebook or script.

### Table 1 — Baseline Replication

```python
import pandas as pd

df = pd.read_csv("results/phase1_baselines.csv")

targets = {
    ("cifar10",  "vit_scratch"):  73.0,
    ("cifar10",  "resnet18"):     95.0,
    ("cifar10",  "sl_vit"):       87.0,
    ("cifar100", "vit_scratch"):  45.0,
    ("cifar100", "resnet18"):     78.0,
    ("cifar100", "sl_vit"):       65.0,
}

rows = []
for (dataset, model), target in targets.items():
    row = df[(df.dataset == dataset) & (df.model == model)]
    achieved = row["val_acc"].values[0] * 100 if len(row) else float("nan")
    rows.append({
        "Model": model, "Dataset": dataset.upper().replace("CIFAR","CIFAR-"),
        "Target (%)": target, "Achieved (%)": round(achieved, 1),
        "Δ (%)": round(achieved - target, 1),
        "Pass?": "✓" if abs(achieved - target) <= 0.5 else "✗"
    })

table1 = pd.DataFrame(rows)
print(table1.to_string(index=False))
table1.to_csv("results/table1_baseline_replication.csv", index=False)
```

---

### Table 2 — Main Results (your paper's central table)

```python
import pandas as pd

df = pd.read_csv("results/phase2_results.csv")
df["val_acc_pct"] = df["val_acc"] * 100

pivot = df.pivot_table(
    values="val_acc_pct", index="variant", columns="dataset", aggfunc="mean"
).round(2)

# Add gain over Variant A
for col in pivot.columns:
    pivot[f"{col}_gain"] = (pivot[col] - pivot.loc["A", col]).round(2)

pivot.columns = ["CIFAR-10 (%)", "CIFAR-100 (%)", "CIFAR-10 Δ vs A", "CIFAR-100 Δ vs A"]
print(pivot.to_string())
pivot.to_csv("results/table2_main_results.csv")
```

Expected shape:
```
         CIFAR-10 (%)  CIFAR-100 (%)  CIFAR-10 Δ  CIFAR-100 Δ
A              72.4           44.1         0.00         0.00
B              76.2           49.3        +3.80        +5.20
C              75.8           48.6        +3.40        +4.50
D              79.1           53.7        +6.70        +9.60   ← D wins on both, gap largest on CIFAR-100
```

---

### Table 3 — OED Range Analysis

```python
import sys, pandas as pd, numpy as np
sys.path.insert(0, "src")
from src.analysis.complexity_oed import range_analysis, L9, FACTOR_NAMES

df = pd.read_csv("results/oed_results.csv")
df = df[df["dataset"] == "cifar100"].copy()

results = {}
for v in ["A", "D"]:
    vdf = df[df["variant"] == v].copy()
    # Sort by experiment order (OED rows 1-9)
    results[v] = (vdf["val_acc"].values[:9] * 100).tolist()

ra = range_analysis(results)
print("=== OED Range Analysis: R = max_level_mean - min_level_mean ===")
print("(Larger R = more sensitive to this factor)")
print(ra[[c for c in ra.columns if c.endswith("_R")]].to_string())
ra.to_csv("results/table3_oed_ranges.csv")
```

Look for: `depth_R` is largest (depth matters most). If Variant D has smaller R values
than Variant A, it means LIFE+SSL makes the model less picky about architecture choices.

---

### Table 5 — Model Parameter Count

```python
import sys, pandas as pd
sys.path.insert(0, "src")
import torch
from src.models.vit_scratch import vit_tiny, vit_base_paper1
from src.models.resnet import resnet18
from src.models.sl_vit import SLViT
from src.models.life_deit import life_deit_tiny

def count_params(m):
    return round(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 2)

rows = [
    ("ResNet-18",       count_params(resnet18(10)),         "CNN — Paper 1 comparison"),
    ("ViT-Scratch",     count_params(vit_base_paper1(10)),  "Paper 1 config, 9.6M"),
    ("ViT-Tiny",        count_params(vit_tiny(10)),         "Variant A/B"),
    ("SL-ViT",          count_params(SLViT(num_classes=10)),"SPT+LSA baseline"),
    ("LIFE-ViT-Tiny",   count_params(life_deit_tiny(10)),   "Variant C/D"),
]
table5 = pd.DataFrame(rows, columns=["Model", "Params (M)", "Notes"])
print(table5.to_string(index=False))
table5.to_csv("results/table5_model_sizes.csv", index=False)
```

---

## Generating Figures Locally

### Figure 1 — Bar chart comparing all 4 variants

```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results/phase2_results.csv")
df["val_acc_pct"] = df["val_acc"] * 100
df["label"] = df["variant"].map({
    "A": "A\nScratch", "B": "B\nSSL-init",
    "C": "C\nLIFE", "D": "D\nSSL+LIFE\n(proposed)"
})
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, dataset in zip(axes, ["cifar10", "cifar100"]):
    sub = df[df.dataset == dataset].sort_values("variant")
    bars = ax.bar(sub["label"], sub["val_acc_pct"], color=colors,
                  edgecolor="white", linewidth=1.5, width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
    ax.set_title(f"{'CIFAR-10' if dataset=='cifar10' else 'CIFAR-100'}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_ylim(sub.val_acc_pct.min() - 6, sub.val_acc_pct.max() + 8)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Combining Architectural + Training Locality for Vision Transformers on Small Data",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("outputs/figure1_main_results.png", dpi=200, bbox_inches="tight")
plt.show()
```

### Figure 2 — OED Factor Effects

```python
import sys; sys.path.insert(0, "src")
from src.analysis.complexity_oed import plot_factor_effects
import pandas as pd

df = pd.read_csv("results/oed_results.csv")
df = df[df["dataset"] == "cifar100"]
results = {
    v: (df[df.variant == v]["val_acc"].values[:9] * 100).tolist()
    for v in ["A", "D"]
}
plot_factor_effects(results, save_path="outputs/figure2_oed_factor_effects.png")
```

---

## Splitting Work Across Multiple Accounts

To run fastest, do this:
```
Account 1 → Phase 1 only (~2h)
Account 2 → Phase 2, Variants A + B  (--variant A then --variant B)
Account 3 → Phase 2, Variants C + D  (--variant C then --variant D)
Account 4 → Phase 3 OED (~2h)
```

Split-variant command example for Account 2's notebook:
```python
!python experiments/phase2_variants.py \
    --variant A \
    --config configs/lite.yaml

!python experiments/phase2_variants.py \
    --variant B \
    --config configs/lite.yaml
```

Account 3 runs C and D the same way. Then merge locally:
```python
import pandas as pd, glob
all_dfs = [pd.read_csv(f) for f in glob.glob("results/phase2_account*.csv")]
merged = pd.concat(all_dfs).drop_duplicates()
merged.to_csv("results/phase2_results.csv", index=False)
```
> Rename the downloaded CSV from each account to `phase2_accountX.csv` before merging.

---

## Final Checklist

```
Data collected:
  [ ] results/phase1_baselines.csv
  [ ] results/phase2_results.csv
  [ ] results/oed_results.csv
  [ ] outputs/phase4/  (CKA pngs + attention map pngs)

Tables built:
  [ ] Table 1 — Baseline replication (±0.5% gate)
  [ ] Table 2 — 4-variant comparison (D wins both datasets)
  [ ] Table 3 — OED range analysis (depth = most sensitive factor)
  [ ] Table 5 — Parameter counts

Figures ready:
  [ ] Figure 1 — Bar chart (4 variants × 2 datasets)
  [ ] Figure 2 — OED factor effects (A vs D sensitivity)
  [ ] Figure 3 — CKA heatmap (outputs/phase4/cka_A_vs_D_cifar100.png)
  [ ] Figure 4 — Attention maps (pick 5 images from attn_A and attn_D folders)

Sanity checks:
  [ ] D > A, B, C on both datasets in Table 2
  [ ] Phase 1 all rows show ✓
  [ ] CKA shows D's lower layers are more similar to ResNet than A's
  [ ] Attention maps: D has local focus in early layers, A does not
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Edit `configs/lite.yaml`: set `batch_size: 32` |
| Variant B/D is slow | Normal — it runs DINO first (50 epochs) then fine-tunes |
| Checkpoint not found in Phase 4 | Run `!ls checkpoints/phase2/` to get exact filenames |
| CKA plot is all zeros | Run `!python -c "import sys; sys.path.insert(0,'src'); from src.models.vit_scratch import vit_tiny; m=vit_tiny(); [print(n) for n,_ in m.named_modules()]"` to verify layer names |
| `phase2_results.csv` missing rows | A variant likely failed silently — re-run that single variant with `--variant X` |
