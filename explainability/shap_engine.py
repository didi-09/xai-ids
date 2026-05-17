"""
SHAP Explainability Engine
- Global: feature importance across all test samples
- Local: per-prediction force/waterfall plot + NLG reasoning
"""
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR


class SHAPEngine:
    def __init__(self, model_path: Optional[Path] = None, feature_names: Optional[list] = None):
        if model_path is None:
            model_path = MODELS_DIR / "rf_binary.pkl"
        self.model = joblib.load(model_path)
        self.feature_names = feature_names or joblib.load(MODELS_DIR / "feature_names.pkl")
        self.explainer = shap.TreeExplainer(self.model)
        self._shap_values_cache = None
        self._X_cache = None

    def compute_shap(self, X: np.ndarray, max_samples: int = 500) -> np.ndarray:
        """Compute SHAP values (capped at max_samples for speed)."""
        X_sub = X[:max_samples]
        print(f"  Computing SHAP values for {len(X_sub)} samples...")
        shap_values = np.array(self.explainer.shap_values(X_sub))
        # Normalize to (n_samples, n_features) for class 1
        if shap_values.ndim == 3:
            if shap_values.shape[0] == len(X_sub):  # (n_samples, n_features, n_classes)
                shap_values = shap_values[:, :, 1]
            else:  # (n_classes, n_samples, n_features)
                shap_values = shap_values[1, :, :]
        self._shap_values_cache = shap_values
        self._X_cache = X_sub
        print("  Done.")
        return shap_values

    def plot_global_importance(self, X: np.ndarray, save_path: Optional[Path] = None):
        """Bar chart of mean |SHAP| per feature."""
        shap_values = self._shap_values_cache if self._X_cache is not None else self.compute_shap(X)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, self._X_cache,
            feature_names=self.feature_names,
            plot_type="bar", show=False
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"  Saved → {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_beeswarm(self, X: np.ndarray, save_path: Optional[Path] = None):
        """Beeswarm: feature impact distribution across samples."""
        shap_values = self._shap_values_cache if self._X_cache is not None else self.compute_shap(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, self._X_cache,
            feature_names=self.feature_names,
            show=False
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        else:
            plt.show()
        plt.close()

    def explain_single(self, x: np.ndarray, idx: int = 0, top_n: int = 5) -> dict:
        """
        Explain a single prediction.
        Returns: {"label": int, "confidence": float, "shap_vals": [...], "reasons": str}
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        pred = self.model.predict(x)[0]
        proba = self.model.predict_proba(x)[0]
        confidence = proba[pred]

        sv = self.explainer.shap_values(x)
        sv = np.array(sv)
        # sv shape: (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
        if sv.ndim == 3:
            if sv.shape[0] == x.shape[0]:  # (n_samples, n_features, n_classes)
                sv_flat = sv[0, :, pred]
            else:  # (n_classes, n_samples, n_features)
                sv_flat = sv[pred, 0, :]
        elif sv.ndim == 2:
            sv_flat = sv[0]
        else:
            sv_flat = sv

        reasons = self._shap_to_nlg(sv_flat, pred, top_n=top_n)
        return {
            "label": int(pred),
            "label_name": "Malicious" if pred == 1 else "Benign",
            "confidence": float(confidence),
            "shap_values": sv_flat.tolist(),
            "feature_names": self.feature_names,
            "reasons": reasons,
        }

    def _shap_to_nlg(self, shap_vals: np.ndarray, pred: int,
                     top_n: int = 5, threshold: float = 0.05) -> str:
        """Convert SHAP values to human-readable reasoning."""
        sv = np.array(shap_vals, dtype=float).flatten()
        pairs = sorted(
            zip(sv.tolist(), self.feature_names),
            key=lambda x: abs(x[0]),
            reverse=True
        )
        label = "Malicious" if pred == 1 else "Benign"
        lines = [f"Traffic classified as: {label}\n\nKey reasons:"]
        count = 0
        for val, feat in pairs:
            if count >= top_n:
                break
            if abs(val) < threshold:
                break
            # For Benign predictions invert so "High X → more Benign" reads naturally
            effective_val = val if pred == 1 else -val
            direction = "High" if effective_val > 0 else "Low"
            target = "malicious" if pred == 1 else "benign"
            lines.append(f"  - {direction} {feat} → more {target} (impact: {abs(val):.4f})")
            count += 1
        return "\n".join(lines)

    def waterfall_html(self, x: np.ndarray) -> str:
        """Return SHAP force plot as HTML string for Streamlit embedding."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sv = self.explainer.shap_values(x)
        if isinstance(sv, list):
            sv = sv[1]
        html = shap.force_plot(
            self.explainer.expected_value if not isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value[1],
            sv[0], x[0],
            feature_names=self.feature_names,
            matplotlib=False
        )
        return shap.getjs() + html.html()


if __name__ == "__main__":
    X_test = np.load(DATA_PROCESSED / "X_test.npy")
    engine = SHAPEngine()
    engine.compute_shap(X_test, max_samples=200)
    engine.plot_global_importance(X_test, save_path=MODELS_DIR / "shap_global.png")
    engine.plot_beeswarm(X_test, save_path=MODELS_DIR / "shap_beeswarm.png")

    result = engine.explain_single(X_test[0])
    print(result["reasons"])
