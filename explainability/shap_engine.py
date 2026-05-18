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
        shap_values = self._normalize_shap(shap_values, X_sub, class_idx=1)
        self._shap_values_cache = shap_values
        self._X_cache = X_sub
        print("  Done.")
        return shap_values

    def _normalize_shap(self, sv: np.ndarray, X: np.ndarray, class_idx: int) -> np.ndarray:
        """Collapse SHAP array to (n_samples, n_features) for the requested class."""
        n = X.shape[0]
        if sv.ndim == 3:
            if sv.shape[0] == n:          # (n_samples, n_features, n_classes)
                return sv[:, :, class_idx]
            else:                          # (n_classes, n_samples, n_features)
                return sv[class_idx, :, :]
        return sv                          # already 2-D

    def plot_global_importance(self, X: np.ndarray, save_path: Optional[Path] = None):
        shap_values = self._shap_values_cache if self._X_cache is not None else self.compute_shap(X)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self._X_cache,
                          feature_names=self.feature_names,
                          plot_type="bar", show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"  Saved → {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_beeswarm(self, X: np.ndarray, save_path: Optional[Path] = None):
        shap_values = self._shap_values_cache if self._X_cache is not None else self.compute_shap(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self._X_cache,
                          feature_names=self.feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        else:
            plt.show()
        plt.close()

    def explain_single(self, x: np.ndarray, top_n: int = 5) -> dict:
        """
        Explain a single prediction.
        Returns: {"label": int, "confidence": float, "shap_values": [...], "reasons": str}
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        pred = int(self.model.predict(x)[0])
        proba = self.model.predict_proba(x)[0]
        confidence = float(proba[pred])

        sv_raw = np.array(self.explainer.shap_values(x))
        # Always explain from the class-1 (malicious) perspective so that
        # positive SHAP = pushes toward malicious for both benign and threat samples.
        sv_flat = self._normalize_shap(sv_raw, x, class_idx=1)[0]

        reasons = self._shap_to_nlg(sv_flat, pred, top_n=top_n)
        return {
            "label": pred,
            "label_name": "Malicious" if pred == 1 else "Benign",
            "confidence": confidence,
            "shap_values": sv_flat.tolist(),
            "feature_names": self.feature_names,
            "reasons": reasons,
        }

    def _shap_to_nlg(self, shap_vals: np.ndarray, pred: int, top_n: int = 5) -> str:
        """Convert class-1 SHAP values to human-readable reasoning."""
        sv = np.array(shap_vals, dtype=float).flatten()
        # Sort by absolute magnitude descending
        pairs = sorted(zip(sv.tolist(), self.feature_names),
                       key=lambda t: abs(t[0]), reverse=True)

        label = "Malicious" if pred == 1 else "Benign"
        lines = [f"Traffic classified as: {label}\n\nKey reasons:"]

        # For threats: positive SHAP = pushes toward malicious → "High X → more malicious"
        # For benign:  negative SHAP = pushes toward benign   → "Low X → less malicious"
        for val, feat in pairs[:top_n]:
            direction = "High" if val > 0 else "Low"
            if pred == 1:
                target = "malicious"
            else:
                # Flip language for benign: a negative class-1 SHAP means
                # this feature is suppressing the malicious signal.
                target = "benign"
                direction = "Low" if val > 0 else "High"
            lines.append(f"  - {direction} {feat} → more {target} (impact: {abs(val):.4f})")

        return "\n".join(lines)

    def waterfall_html(self, x: np.ndarray) -> str:
        """Return SHAP force plot as a self-contained HTML string."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sv = np.array(self.explainer.shap_values(x))
        sv_flat = self._normalize_shap(sv, x, class_idx=1)
        base = self.explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(base[1])
        plot = shap.force_plot(float(base), sv_flat[0], x[0],
                               feature_names=self.feature_names,
                               matplotlib=False)
        import io
        buf = io.StringIO()
        shap.save_html(buf, plot)
        return buf.getvalue()


if __name__ == "__main__":
    X_test = np.load(DATA_PROCESSED / "X_test.npy")
    engine = SHAPEngine()
    engine.compute_shap(X_test, max_samples=200)
    engine.plot_global_importance(X_test, save_path=MODELS_DIR / "shap_global.png")
    engine.plot_beeswarm(X_test, save_path=MODELS_DIR / "shap_beeswarm.png")

    result = engine.explain_single(X_test[0])
    print(result["reasons"])
