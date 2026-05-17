"""
Phase 1 — Exploratory Data Analysis
Run as a script: python notebooks/01_eda.py
Or open in Jupyter: jupyter lab notebooks/01_eda.ipynb (generated from this)
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATASET_PATH, LABEL_COL, ATTACK_COL, DATA_SAMPLES

sns.set_theme(style="darkgrid")

print("=" * 60)
print("PHASE 1 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\n[1] Loading dataset from:\n    {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH, low_memory=False)
print(f"    Shape: {df.shape}")

# ── Basic info ─────────────────────────────────────────────────────────────
print("\n[2] Column types:")
print(df.dtypes.value_counts().to_string())

print("\n[3] Missing values (top 10):")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0].head(10).to_string())

print(f"\n[4] Duplicate rows: {df.duplicated().sum()}")

# ── Label distribution ─────────────────────────────────────────────────────
print(f"\n[5] Label distribution ({LABEL_COL}):")
print(df[LABEL_COL].value_counts().to_string())

if ATTACK_COL in df.columns:
    print(f"\n[6] Attack type distribution ({ATTACK_COL}):")
    print(df[ATTACK_COL].value_counts().to_string())

# ── Statistical summary ────────────────────────────────────────────────────
print("\n[7] Numerical feature summary (first 5 cols):")
num_cols = df.select_dtypes(include=[np.number]).columns[:5]
print(df[num_cols].describe().to_string())

# ── Save small sample for fast testing ───────────────────────────────────
DATA_SAMPLES.mkdir(parents=True, exist_ok=True)
sample = df.sample(n=min(10000, len(df)), random_state=42)
sample_path = DATA_SAMPLES / "sample_10k.csv"
sample.to_csv(sample_path, index=False)
print(f"\n[8] Saved 10k sample → {sample_path}")

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
vc = df[LABEL_COL].value_counts()
axes[0].bar(["Benign", "Malicious"], vc.values, color=["steelblue", "tomato"])
axes[0].set_title("Binary Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + vc.max() * 0.01, f"{v:,}", ha="center")

# Attack type distribution
if ATTACK_COL in df.columns:
    atk_vc = df[ATTACK_COL].value_counts()
    axes[1].barh(atk_vc.index[:10], atk_vc.values[:10], color="coral")
    axes[1].set_title("Top 10 Attack Types")
    axes[1].set_xlabel("Count")

plt.tight_layout()
plot_path = DATA_SAMPLES / "eda_class_dist.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"[9] Saved class distribution plot → {plot_path}")
plt.close()

# Correlation heatmap (numeric only, top 15 cols)
num_df = df.select_dtypes(include=[np.number]).iloc[:, :15]
fig, ax = plt.subplots(figsize=(12, 10))
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
            center=0, ax=ax, linewidths=0.3)
ax.set_title("Feature Correlation Heatmap (top 15 features)")
plt.tight_layout()
corr_path = DATA_SAMPLES / "eda_correlation.png"
plt.savefig(corr_path, dpi=150, bbox_inches="tight")
print(f"[10] Saved correlation heatmap → {corr_path}")
plt.close()

print("\n✓ EDA complete.")
print(f"   All outputs in: {DATA_SAMPLES}")
