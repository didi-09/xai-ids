"""Generate architectural diagrams for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

OUT = Path(__file__).parent.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared helpers ────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel="", color="#1565C0", fontsize=10,
        text_color="white", radius=0.04, alpha=1.0):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad={radius}",
                           linewidth=1.2, edgecolor="white",
                           facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(fancy)
    if sublabel:
        ax.text(x, y + 0.055, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - 0.07, sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color=text_color, alpha=0.88, zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", color="#455A64", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.02, my, label, ha="left", va="center",
                fontsize=7.5, color="#37474F", style="italic")

def horiz_arrow(ax, x1, x2, y, label="", color="#455A64", lw=1.8):
    arrow(ax, x1, y, x2, y, label=label, color=color, lw=lw)

def vert_arrow(ax, x, y1, y2, label="", color="#455A64", lw=1.8):
    arrow(ax, x, y1, x, y2, label=label, color=color, lw=lw)


# ── Diagram 1: Full System Architecture ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)
ax.axis("off")
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FAFAFA")

ax.text(6.5, 5.15, "XAI-IDS — Full System Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1A237E")

# Row 1: input
box(ax, 1.2, 4.2, 1.8, 0.65, "Network Traffic", "Raw PCAP / NetFlow", "#37474F", fontsize=9)

# Row 1 → Row 2
horiz_arrow(ax, 2.1, 2.9, 4.2)

# Row 2: preprocessing
box(ax, 3.8, 4.2, 1.6, 0.65, "Preprocessing", "Scale · Select · Clean", "#1565C0", fontsize=9)

horiz_arrow(ax, 4.6, 5.4, 4.2)

# Row 3: model
box(ax, 6.3, 4.2, 1.6, 0.65, "Random Forest", "Binary · Multiclass", "#6A1B9A", fontsize=9)

horiz_arrow(ax, 7.1, 7.9, 4.2)

# Row 4: SHAP
box(ax, 8.8, 4.2, 1.6, 0.65, "SHAP Engine", "TreeSHAP · Global/Local", "#00695C", fontsize=9)

horiz_arrow(ax, 9.6, 10.4, 4.2)

# Row 5: NLG
box(ax, 11.3, 4.2, 1.6, 0.65, "NLG Module", "Human-readable reasons", "#E65100", fontsize=9)

# Outputs: drop down from NLG
vert_arrow(ax, 11.3, 3.875, 3.25)

# Output boxes
box(ax, 9.5, 2.7, 1.8, 0.65, "Streamlit\nDashboard", "", "#1976D2", fontsize=9)
box(ax, 11.3, 2.7, 1.8, 0.65, "FastAPI\nREST Endpoint", "", "#1976D2", fontsize=9)
box(ax, 13.1, 2.7, 1.6, 0.65, "Alert /\nSIEM Export", "", "#1976D2", fontsize=9)

# arrows to each output
arrow(ax, 11.3, 3.25, 9.5, 3.035)
arrow(ax, 11.3, 3.25, 11.3, 3.035)
arrow(ax, 11.3, 3.25, 13.1, 3.035)

# ── Lower section: data detail ─────────────────────────────────────────────
ax.axhline(2.2, color="#CFD8DC", lw=0.8, xmin=0.01, xmax=0.99)
ax.text(0.3, 1.95, "Data flow:", fontsize=8.5, color="#546E7A", fontweight="bold")

stages = [
    (1.2,  "Raw CSV\n2.36M × 55"),
    (3.8,  "Scaled\n1.88M × 32"),
    (6.3,  "Prediction\n+ Probability"),
    (8.8,  "SHAP Values\n32 floats"),
    (11.3, "Ranked\nReasons"),
]
for x, lbl in stages:
    ax.text(x, 1.65, lbl, ha="center", va="center", fontsize=8,
            color="#37474F", bbox=dict(boxstyle="round,pad=0.25",
            facecolor="#ECEFF1", edgecolor="#B0BEC5", linewidth=0.8))

for i in range(len(stages)-1):
    x1 = stages[i][0] + 0.72
    x2 = stages[i+1][0] - 0.72
    ax.annotate("", xy=(x2, 1.65), xytext=(x1, 1.65),
                arrowprops=dict(arrowstyle="-|>", color="#78909C", lw=1.2,
                                mutation_scale=10), zorder=2)

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_system.png", dpi=180, bbox_inches="tight",
            facecolor="#FAFAFA")
plt.close()
print("✓ arch_system.png")


# ── Diagram 2: Preprocessing Pipeline ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.8)
ax.axis("off")
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FAFAFA")

ax.text(6, 4.45, "Preprocessing Pipeline", ha="center", fontsize=13,
        fontweight="bold", color="#1A237E")

steps = [
    (0.9,  "Load CSV",        "2,365,424 × 55",  "#37474F"),
    (2.65, "Deduplicate",     "−14,815 rows",     "#1565C0"),
    (4.4,  "Impute / Fix inf","median fill",      "#1565C0"),
    (6.15, "Drop columns",    "IP, timestamp,\nsparse cols", "#6A1B9A"),
    (7.9,  "Corr. Pruning",   "|r|>0.95 → drop\n43→32 features","#00695C"),
    (9.65, "RobustScaler",    "fit on train\ntransform both",  "#E65100"),
    (11.4, "Train/Test\nSplit","80% / 20%\nstratified",        "#B71C1C"),
]

for x, title, sub, col in steps:
    box(ax, x, 2.5, 1.45, 1.0, title, sub, col, fontsize=8.5)

for i in range(len(steps)-1):
    x1 = steps[i][0] + 0.725
    x2 = steps[i+1][0] - 0.725
    horiz_arrow(ax, x1, x2, 2.5)

# Output annotation
ax.text(6, 1.35, "Output:  X_train (1,880,487 × 32)  ·  X_test (470,122 × 32)  ·  scaler.pkl  ·  feature_names.pkl",
        ha="center", fontsize=9, color="#37474F",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9",
                  edgecolor="#81C784", linewidth=1))

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_preprocessing.png", dpi=180, bbox_inches="tight",
            facecolor="#FAFAFA")
plt.close()
print("✓ arch_preprocessing.png")


# ── Diagram 3: Explainability Pipeline ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.8)
ax.axis("off")
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FAFAFA")

ax.text(6, 5.45, "Explainability Pipeline", ha="center", fontsize=13,
        fontweight="bold", color="#1A237E")

# Top row: input → model → SHAP raw
box(ax, 1.2, 4.3, 1.8, 0.75, "Flow Record", "32 scaled features", "#37474F", fontsize=9)
horiz_arrow(ax, 2.1, 2.85, 4.3)
box(ax, 3.75, 4.3, 1.6, 0.75, "Random Forest\nPredict", "label + probability", "#6A1B9A", fontsize=9)
horiz_arrow(ax, 4.55, 5.35, 4.3)
box(ax, 6.2, 4.3, 1.7, 0.75, "TreeSHAP", "φ_j per feature\n(32 values)", "#00695C", fontsize=9)

# Split: global vs local
vert_arrow(ax, 6.2, 3.925, 3.3)
arrow(ax, 6.2, 3.925, 9.2, 3.3)

# Global branch (left)
box(ax, 4.5, 2.9, 2.5, 0.7, "Global Explanation",
    "mean|φ_j| over N samples", "#1565C0", fontsize=9)
vert_arrow(ax, 4.5, 2.55, 1.85)
box(ax, 4.5, 1.5, 2.5, 0.65, "Feature Importance\nBar Chart", "", "#1976D2", fontsize=9)

# Local branch (right)
box(ax, 9.2, 2.9, 2.5, 0.7, "Local Explanation",
    "φ_j for single flow", "#E65100", fontsize=9)
vert_arrow(ax, 9.2, 2.55, 1.85)
box(ax, 9.2, 1.5, 2.5, 0.65, "NLG + Force Plot\n(per alert)", "", "#EF6C00", fontsize=9)

# NLG output box
ax.text(6, 0.75,
        '"Traffic classified as: Malicious  |  High MIN_TTL → more malicious (impact: 0.254)"',
        ha="center", fontsize=8.5, color="#1B5E20", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                  edgecolor="#66BB6A", linewidth=1))
arrow(ax, 4.5, 1.175, 6.0, 0.95)
arrow(ax, 9.2, 1.175, 7.0, 0.95)

# Labels
ax.text(4.5, 3.65, "Global\nbranch", ha="center", fontsize=8,
        color="#1565C0", style="italic")
ax.text(9.2, 3.65, "Local\nbranch", ha="center", fontsize=8,
        color="#E65100", style="italic")

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_explainability.png", dpi=180, bbox_inches="tight",
            facecolor="#FAFAFA")
plt.close()
print("✓ arch_explainability.png")


# ── Diagram 4: Training Architecture ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)
ax.axis("off")
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("#FAFAFA")

ax.text(5.5, 5.15, "Model Training Architecture", ha="center", fontsize=13,
        fontweight="bold", color="#1A237E")

# Shared input
box(ax, 5.5, 4.3, 2.6, 0.7, "Preprocessed Data",
    "X_train 1.88M×32 · y_train", "#37474F", fontsize=9)

# Split into two branches
arrow(ax, 4.2, 3.95, 2.5, 3.35)
arrow(ax, 6.8, 3.95, 8.5, 3.35)

# Left branch: Binary
box(ax, 2.5, 3.0, 3.2, 0.6, "Binary Classification", "Benign vs Malicious", "#1565C0", fontsize=9)
vert_arrow(ax, 2.5, 2.7, 2.1)
box(ax, 1.1, 1.75, 2.0, 0.7, "RF Full\nN=100, d=15", "class_weight=balanced", "#1976D2", fontsize=8.5)
box(ax, 3.3, 1.75, 2.0, 0.7, "RF Edge\nN=25, d=8", "IoT deployment", "#42A5F5", fontsize=8.5)
arrow(ax, 2.5, 2.7, 1.3, 2.1)
arrow(ax, 2.5, 2.7, 3.1, 2.1)

# Binary metrics
ax.text(2.2, 1.0,
        "F1=99.94%  AUC=1.00  FNR=0.02%\nEdge: 11,055 flows/sec · 0.25 MB",
        ha="center", fontsize=8, color="#0D47A1",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                  edgecolor="#90CAF9", lw=0.9))

# Right branch: Multiclass
box(ax, 8.5, 3.0, 3.2, 0.6, "Multiclass Classification", "10 attack categories", "#6A1B9A", fontsize=9)
vert_arrow(ax, 8.5, 2.7, 2.1)
box(ax, 7.3, 1.75, 2.0, 0.7, "RF\nMulticlass", "N=100, d=15", "#8E24AA", fontsize=8.5)
box(ax, 9.7, 1.75, 2.0, 0.7, "XGBoost\nMulticlass", "lr=0.1, d=10", "#AB47BC", fontsize=8.5)
arrow(ax, 8.5, 2.7, 7.5, 2.1)
arrow(ax, 8.5, 2.7, 9.5, 2.1)

ax.text(8.8, 1.0,
        "Acc=98.43%  Weighted F1=99.21%\nMacro F1=63.65% (imbalanced rare classes)",
        ha="center", fontsize=8, color="#4A148C",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5",
                  edgecolor="#CE93D8", lw=0.9))

# Divider
ax.axvline(5.5, color="#CFD8DC", lw=0.8, ymin=0.1, ymax=0.75, linestyle="--")

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "arch_training.png", dpi=180, bbox_inches="tight",
            facecolor="#FAFAFA")
plt.close()
print("✓ arch_training.png")

print(f"\nAll diagrams saved → {OUT}")
