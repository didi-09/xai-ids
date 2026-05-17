"""
Binary classifier: Benign (0) vs Malicious (1)
Run: python models/train_binary.py
Expects preprocessed data in data/processed/
"""
import numpy as np
import joblib
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH
from utils.metrics import evaluate_binary, save_metrics, print_summary
from preprocessing.imbalance import apply_smote


def load_data():
    X_train = np.load(DATA_PROCESSED / "X_train.npy")
    X_test = np.load(DATA_PROCESSED / "X_test.npy")
    y_train = np.load(DATA_PROCESSED / "y_train.npy")
    y_test = np.load(DATA_PROCESSED / "y_test.npy")
    return X_train, X_test, y_train, y_test


def train(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, use_smote=True, edge_mode=False):
    print("=== Binary RF Training ===")
    X_train, X_test, y_train, y_test = load_data()
    feature_names = joblib.load(DATA_PROCESSED / "feature_names.pkl")
    print(f"  Features: {len(feature_names)} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    if edge_mode:
        n_estimators = 25
        max_depth = 8
        print(f"  Edge mode: n_estimators={n_estimators}, max_depth={max_depth}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("  Training...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_binary(y_test, y_pred, y_proba)
    print_summary(metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_edge" if edge_mode else ""
    model_path = MODELS_DIR / f"rf_binary{suffix}.pkl"
    joblib.dump(model, model_path)
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    save_metrics(metrics, MODELS_DIR / f"metrics_binary{suffix}.json")
    print(f"  Model saved → {model_path}")

    return model, metrics, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", action="store_true", help="Train lightweight edge model")
    parser.add_argument("--no-smote", action="store_true", help="Skip SMOTE")
    args = parser.parse_args()
    train(use_smote=not args.no_smote, edge_mode=args.edge)
