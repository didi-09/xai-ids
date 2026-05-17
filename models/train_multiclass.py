"""
Multiclass classifier: per attack-type classification
Run after binary model is validated.
"""
import numpy as np
import joblib
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH
from utils.metrics import evaluate_multiclass, save_metrics, print_summary


def load_attack_labels():
    y_attack_test = np.load(DATA_PROCESSED / "y_attack_test.npy", allow_pickle=True)
    return y_attack_test


def train_multiclass(model_type="rf"):
    print(f"=== Multiclass Training ({model_type.upper()}) ===")
    X_train = np.load(DATA_PROCESSED / "X_train.npy")
    X_test = np.load(DATA_PROCESSED / "X_test.npy")

    # For multiclass we use attack string labels stored separately
    y_train_path = DATA_PROCESSED / "y_attack_train.npy"
    y_test_path = DATA_PROCESSED / "y_attack_test.npy"

    if not y_train_path.exists():
        print("  Attack labels not available — run pipeline.py first with full dataset")
        return

    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    joblib.dump(le, MODELS_DIR / "label_encoder_multiclass.pkl")

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=10,
            learning_rate=0.1, eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_STATE
        )

    print("  Training...")
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)

    metrics = evaluate_multiclass(y_test_enc, y_pred, labels=list(range(len(le.classes_))))
    metrics["class_names"] = le.classes_.tolist()
    print_summary(metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_type}_multiclass.pkl"
    joblib.dump(model, model_path)
    save_metrics(metrics, MODELS_DIR / f"metrics_multiclass_{model_type}.json")
    print(f"  Model saved → {model_path}")

    return model, metrics, le


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    args = parser.parse_args()
    train_multiclass(args.model)
