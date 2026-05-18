"""Generate architectural diagrams for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, label, sublabel="", color="#1565C0", fontsize=10,
        text_color="white", radius=0.03, alpha=1.0):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad={radius}",
                           linewidth=1.2, edgecolor="white",
                           facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(fancy)
    if sublabel:
        ax.text(x, y + 0.06, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - 0.08, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, alpha=0.9, zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#455A64", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14), zorder=2)


def harrow(ax, x1, x2, y, color="#455A64", lw=1.8):
    arrow(ax, x1, y, x2, y, color=color, lw=lw)


def varrow(ax, x, y1, y2, color="#455A64", lw=1.8):
    arrow(ax, x, y1, x, y2, color=color, lw=lw)


# ── Diagram 1: Full System Architecture ──────────────────────────────────────

W, H = 15, 6.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_facecolor("#F8FAFB")
fig.patch.set_facecolor("#F8FAFB")

ax.text(W/2, 6.15, "XAI-IDS — Full System Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

# Pipeline row (y=4.6)
pipe_y = 4.6
bw, bh = 2.0, 0.85
centers = [1.3, 3.6, 5.9, 8.2, 10.5, 12.8]
labels = [
    ("Network Traffic",   "Raw PCAP / NetFlow",      "#37474F"),
    ("Preprocessing",     "Scale · Select · Clean",  "#1565C0"),
    ("Random Forest",     "Binary · Multiclass",     "#6A1B9A"),
    ("SHAP Engine",       "TreeSHAP · Global/Local", "#00695C"),
    ("NLG Module",        "Plain-English reasons",   "#E65100"),
    ("Output Layer",      "Dashboard · API · SIEM",  "#1976D2"),
]
for (cx, (lbl, sub, col)) in zip(centers, labels):
    box(ax, cx, pipe_y, bw, bh, lbl, sub, col, fontsize=9)

for i in range(len(centers) - 1):
    harrow(ax, centers[i] + bw/2 + 0.05, centers[i+1] - bw/2 - 0.05, pipe_y)

# Output sub-boxes (y=2.9)
out_y = 2.9
out_boxes = [(10.5, "Streamlit\nDashboard", "#1E88E5"),
             (12.2, "FastAPI REST\nEndpoint",  "#1E88E5"),
             (13.9, "SIEM /\nAlert Export",    "#1E88E5")]
for ox, lbl, col in out_boxes:
    box(ax, ox, out_y, 1.5, 0.75, lbl, "", col, fontsize=8.5)
    arrow(ax, 12.8, pipe_y - bh/2 - 0.05, ox, out_y + 0.375)

# Data-flow strip (y=1.7)
ax.axhline(2.1, color="#CFD8DC", lw=0.8, xmin=0.01, xmax=0.99)
ax.text(0.25, 1.85, "Data flow:", fontsize=8.5, color="#546E7A", fontweight="bold")

data_stages = [
    (1.3,  "Raw CSV\n2.36M × 55"),
    (3.6,  "Scaled\n1.88M × 32"),
    (5.9,  "Prediction\n+ Proba"),
    (8.2,  "SHAP\n32 floats"),
    (10.5, "Ranked\nReasons"),
]
for sx, lbl in data_stages:
    ax.text(sx, 1.4, lbl, ha="center", va="center", fontsize=8.5, color="#37474F",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECEFF1",
                      edgecolor="#B0BEC5", linewidth=0.8))

for i in range(len(data_stages) - 1):
    x1 = data_stages[i][0] + 0.72
    x2 = data_stages[i+1][0] - 0.72
    ax.annotate("", xy=(x2, 1.4), xytext=(x1, 1.4),
                arrowprops=dict(arrowstyle="-|>", color="#78909C", lw=1.1,
                                mutation_scale=10), zorder=2)

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_system.png", dpi=180, bbox_inches="tight", facecolor="#F8FAFB")
plt.close()
print("✓ arch_system.png")


# ── Diagram 2: Preprocessing Pipeline ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 5.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.2)
ax.axis("off")
ax.set_facecolor("#F8FAFB")
fig.patch.set_facecolor("#F8FAFB")

ax.text(7, 4.85, "Preprocessing Pipeline",
        ha="center", fontsize=13, fontweight="bold", color="#1A237E")

steps = [
    (1.0,  "Load CSV",        "2,365,424 × 55",      "#37474F"),
    (3.0,  "Deduplicate",     "−14,815 rows",         "#1565C0"),
    (5.0,  "Impute / Fix∞",   "median fill",          "#1565C0"),
    (7.0,  "Drop Columns",    "IP · timestamp\nsparse","#6A1B9A"),
    (9.0,  "Corr. Pruning",   "|r|>0.95\n43→32 feats","#00695C"),
    (11.0, "RobustScaler",    "fit train\ntransform both","#E65100"),
    (13.0, "Train/Test Split","80%/20%\nstratified",  "#B71C1C"),
]
bw, bh = 1.7, 1.1

for x, title, sub, col in steps:
    box(ax, x, 2.9, bw, bh, title, sub, col, fontsize=8.5)

for i in range(len(steps) - 1):
    harrow(ax, steps[i][0] + bw/2 + 0.05, steps[i+1][0] - bw/2 - 0.05, 2.9)

ax.text(7, 1.5,
        "Output:  X_train (1,880,487 × 32)   ·   X_test (470,122 × 32)"
        "   ·   scaler.pkl   ·   feature_names.pkl",
        ha="center", fontsize=9.5, color="#37474F",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                  edgecolor="#81C784", linewidth=1.2))

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_preprocessing.png", dpi=180, bbox_inches="tight",
            facecolor="#F8FAFB")
plt.close()
print("✓ arch_preprocessing.png")


# ── Diagram 3: Explainability Pipeline ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis("off")
ax.set_facecolor("#F8FAFB")
fig.patch.set_facecolor("#F8FAFB")

ax.text(6.5, 6.15, "Explainability Pipeline",
        ha="center", fontsize=13, fontweight="bold", color="#1A237E")

# Top row
top_y = 5.0
for cx, lbl, sub, col in [
    (1.5,  "Flow Record",        "32 scaled features",    "#37474F"),
    (4.0,  "Random Forest",      "label + probability",   "#6A1B9A"),
    (6.5,  "TreeSHAP",           "φⱼ for each feature",  "#00695C"),
]:
    box(ax, cx, top_y, 2.2, 0.85, lbl, sub, col, fontsize=9)

harrow(ax, 1.5 + 1.1 + 0.05, 4.0 - 1.1 - 0.05, top_y)
harrow(ax, 4.0 + 1.1 + 0.05, 6.5 - 1.1 - 0.05, top_y)

# Split from TreeSHAP down to two branches
split_y = top_y - 0.425
branch_y = 3.0

arrow(ax, 6.5, split_y, 3.5, branch_y + 0.425)
varrow(ax, 6.5, split_y, branch_y + 0.425)
arrow(ax, 6.5, split_y, 9.5, branch_y + 0.425)

# Global branch
box(ax, 3.5, branch_y, 2.8, 0.85, "Global Explanation",
    "mean |φⱼ| over N samples", "#1565C0", fontsize=9)
ax.text(3.5, branch_y + 0.65, "Global branch", ha="center", fontsize=8,
        color="#1565C0", style="italic")

# Local branch
box(ax, 9.5, branch_y, 2.8, 0.85, "Local Explanation",
    "φⱼ for single prediction", "#E65100", fontsize=9)
ax.text(9.5, branch_y + 0.65, "Local branch", ha="center", fontsize=8,
        color="#E65100", style="italic")

out_y = 1.6
varrow(ax, 3.5, branch_y - 0.425, out_y + 0.425)
varrow(ax, 9.5, branch_y - 0.425, out_y + 0.425)

box(ax, 3.5, out_y, 2.8, 0.85, "Feature Importance\nBar Chart", "", "#1976D2", fontsize=9)
box(ax, 9.5, out_y, 2.8, 0.85, "NLG Reasoning\n+ Force Plot", "", "#EF6C00", fontsize=9)

# NLG example
nlg_y = 0.52
arrow(ax, 3.5,  out_y - 0.425, 5.5, nlg_y + 0.2)
arrow(ax, 9.5, out_y - 0.425, 7.5, nlg_y + 0.2)
ax.text(6.5, nlg_y,
        '"Traffic classified as: Malicious  |  High IN_PKTS → more malicious (impact: 0.423)"',
        ha="center", fontsize=8.5, color="#1B5E20", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                  edgecolor="#66BB6A", linewidth=1.2))

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_explainability.png", dpi=180, bbox_inches="tight",
            facecolor="#F8FAFB")
plt.close()
print("✓ arch_explainability.png")


# ── Diagram 4: Training Architecture ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 6.8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.8)
ax.axis("off")
ax.set_facecolor("#F8FAFB")
fig.patch.set_facecolor("#F8FAFB")

ax.text(6.5, 6.45, "Model Training Architecture",
        ha="center", fontsize=13, fontweight="bold", color="#1A237E")

# Shared input
box(ax, 6.5, 5.5, 3.2, 0.85, "Preprocessed Dataset",
    "X_train: 1,880,487 × 32   ·   y_train", "#37474F", fontsize=9)

# Branch arrows
arrow(ax, 4.9, 5.075, 3.0, 4.475)
arrow(ax, 8.1, 5.075, 10.0, 4.475)

# ── Left: Binary branch ──────────────────────────────────────────────────────
box(ax, 3.0, 4.1, 3.5, 0.75, "Binary Classification",
    "Benign vs. Malicious", "#1565C0", fontsize=9)

arrow(ax, 1.8, 3.725, 1.5, 3.2)
arrow(ax, 4.2, 3.725, 4.5, 3.2)

box(ax, 1.5, 2.85, 2.5, 0.7, "RF Full\nN=100 · depth=15",
    "class_weight=balanced", "#1976D2", fontsize=8.5)
box(ax, 4.5, 2.85, 2.5, 0.7, "RF Edge\nN=25 · depth=8",
    "IoT-optimized", "#42A5F5", fontsize=8.5)

ax.text(3.0, 1.85,
        "Binary F1 = 99.94%   AUC = 1.00   FNR = 0.02%\n"
        "Edge: 11,055 flows/sec · 0.25 MB model",
        ha="center", fontsize=8.5, color="#0D47A1",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E3F2FD",
                  edgecolor="#90CAF9", lw=1.0))

# ── Right: Multiclass branch ─────────────────────────────────────────────────
box(ax, 10.0, 4.1, 3.5, 0.75, "Multiclass Classification",
    "10 attack categories", "#6A1B9A", fontsize=9)

arrow(ax, 8.8, 3.725, 8.5, 3.2)
arrow(ax, 11.2, 3.725, 11.5, 3.2)

box(ax, 8.5, 2.85, 2.5, 0.7, "RF\nMulticlass",
    "N=100 · depth=15", "#8E24AA", fontsize=8.5)
box(ax, 11.5, 2.85, 2.5, 0.7, "XGBoost\nMulticlass",
    "lr=0.1 · depth=10", "#AB47BC", fontsize=8.5)

ax.text(10.0, 1.85,
        "Acc = 98.43%   Weighted F1 = 99.21%\n"
        "Macro F1 = 63.65%  (rare classes limited by samples)",
        ha="center", fontsize=8.5, color="#4A148C",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F3E5F5",
                  edgecolor="#CE93D8", lw=1.0))

# Divider
ax.axvline(6.5, color="#CFD8DC", lw=1.0, ymin=0.22, ymax=0.82, linestyle="--")
ax.text(6.5, 1.1, "Binary", ha="center", fontsize=8.5, color="#1565C0",
        style="italic", fontweight="bold")
ax.text(6.5, 0.7, "←  ─  ─  ─  ─  ─  ─  ─  ─  →", ha="center",
        fontsize=8, color="#90A4AE")

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_training.png", dpi=180, bbox_inches="tight",
            facecolor="#F8FAFB")
plt.close()
print("✓ arch_training.png")

print(f"\nAll diagrams saved → {OUT}")
