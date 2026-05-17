import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)


def evaluate_binary(y_true, y_pred, y_proba=None) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    result = {
        "accuracy": report["accuracy"],
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": fnr,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
    return result


def evaluate_multiclass(y_true, y_pred, labels=None) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved → {path}")


def print_summary(metrics: dict):
    print(f"  Accuracy : {metrics.get('accuracy', 0):.4f}")
    print(f"  F1       : {metrics.get('f1', metrics.get('macro_f1', 0)):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall   : {metrics.get('recall', 0):.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    if "false_negative_rate" in metrics:
        print(f"  FNR      : {metrics['false_negative_rate']:.4f}")
