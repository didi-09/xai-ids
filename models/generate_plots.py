"""Generate all evaluation plots for the report."""
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, classification_report
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR

sns.set_theme(style="whitegrid", palette="muted")
REPORT = Path(__file__).parent.parent / "report" / "figures"
REPORT.mkdir(parents=True, exist_ok=True)

X_test  = np.load(DATA_PROCESSED / "X_test.npy")
y_test  = np.load(DATA_PROCESSED / "y_test.npy")
y_attack_test = np.load(DATA_PROCESSED / "y_attack_test.npy", allow_pickle=True)

model_full = joblib.load(MODELS_DIR / "rf_binary.pkl")
model_edge = joblib.load(MODELS_DIR / "rf_binary_edge.pkl")
feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

y_pred_full  = model_full.predict(X_test)
y_proba_full = model_full.predict_proba(X_test)[:, 1]
y_pred_edge  = model_edge.predict(X_test)
y_proba_edge = model_edge.predict_proba(X_test)[:, 1]

# ── 1. Confusion Matrix (full model) ─────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_full)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"],
            cbar_kws={"label": "Count"})
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — Binary RF (N=100)", fontsize=12)
plt.tight_layout()
plt.savefig(REPORT / "confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ confusion_matrix.png")

# ── 2. ROC Curves (full vs edge) ─────────────────────────────────────────
fpr_f, tpr_f, _ = roc_curve(y_test, y_proba_full)
fpr_e, tpr_e, _ = roc_curve(y_test, y_proba_edge)
auc_f = auc(fpr_f, tpr_f)
auc_e = auc(fpr_e, tpr_e)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr_f, tpr_f, lw=2, label=f"RF Full (AUC = {auc_f:.4f})", color="#2196F3")
ax.plot(fpr_e, tpr_e, lw=2, label=f"RF Edge (AUC = {auc_e:.4f})", color="#FF5722", linestyle="--")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Binary Classification", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(REPORT / "roc_curve.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ roc_curve.png")

# ── 3. Per-class F1 — Multiclass RF ──────────────────────────────────────
le = joblib.load(MODELS_DIR / "label_encoder_multiclass.pkl")
mc_model = joblib.load(MODELS_DIR / "rf_multiclass.pkl")
y_attack_enc = le.transform(y_attack_test)
y_mc_pred = mc_model.predict(X_test)

report = classification_report(y_attack_enc, y_mc_pred,
                               target_names=le.classes_, output_dict=True, zero_division=0)
classes = le.classes_
f1s = [report[c]["f1-score"] for c in classes]
supports = [report[c]["support"] for c in classes]

colors = ["#4CAF50" if f >= 0.9 else "#FF9800" if f >= 0.6 else "#F44336" for f in f1s]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(classes, f1s, color=colors, edgecolor="white", height=0.6)
for bar, f1, sup in zip(bars, f1s, supports):
    ax.text(min(f1 + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{f1:.2f}  (n={sup:,})", va="center", fontsize=9)
ax.set_xlim(0, 1.12)
ax.set_xlabel("F1-Score", fontsize=11)
ax.set_title("Per-Class F1-Score — Multiclass RF", fontsize=12)
ax.axvline(0.9, color="gray", linestyle="--", lw=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(REPORT / "multiclass_f1.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ multiclass_f1.png")

# ── 4. Feature Importance (RF built-in, ranked) ───────────────────────────
importances = model_full.feature_importances_
idx = np.argsort(importances)[-15:]  # top 15
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([feature_names[i] for i in idx], importances[idx],
        color="#5C6BC0", edgecolor="white")
ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
ax.set_title("Top-15 Feature Importances — Random Forest", fontsize=12)
plt.tight_layout()
plt.savefig(REPORT / "feature_importance.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ feature_importance.png")

# ── 5. Throughput vs Accuracy tradeoff (edge vs full) ────────────────────
bench = {"RF Full\n(N=100, d=15)": (5519, 99.94),
         "RF Edge\n(N=25, d=8)":  (11055, 99.91)}

fig, ax = plt.subplots(figsize=(5.5, 4))
for label, (tput, f1) in bench.items():
    ax.scatter(tput, f1, s=200, zorder=5,
               color="#2196F3" if "Full" in label else "#FF5722")
    ax.annotate(label, (tput, f1), textcoords="offset points",
                xytext=(8, -12), fontsize=9)
ax.set_xlabel("Throughput (flows / sec)", fontsize=11)
ax.set_ylabel("F1-Score (%)", fontsize=11)
ax.set_title("Throughput vs. F1-Score Tradeoff", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
ax.set_xlim(0, 14000)
ax.set_ylim(99.85, 99.98)
plt.tight_layout()
plt.savefig(REPORT / "throughput_tradeoff.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ throughput_tradeoff.png")

# ── 6. Copy SHAP plots into report/figures ────────────────────────────────
import shutil
for name in ["shap_global.png", "shap_beeswarm.png"]:
    src = MODELS_DIR / name
    if src.exists():
        shutil.copy(src, REPORT / name)
        print(f"✓ {name} copied")

# ── 7. Copy EDA plots ─────────────────────────────────────────────────────
DATA_SAMPLES = Path(__file__).parent.parent / "data" / "samples"
for name in ["eda_class_dist.png", "eda_correlation.png"]:
    src = DATA_SAMPLES / name
    if src.exists():
        shutil.copy(src, REPORT / name)
        print(f"✓ {name} copied")

print(f"\nAll figures saved → {REPORT}")
