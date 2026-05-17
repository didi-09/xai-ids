import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import (
    DATASET_PATH, DATA_PROCESSED, LABEL_COL, ATTACK_COL,
    TARGET_FEATURES, RANDOM_STATE, TEST_SIZE, DROP_COLS
)


def load_raw(path: Path = DATASET_PATH) -> pd.DataFrame:
    print(f"Loading dataset from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    return df


def drop_junk(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(thresh=int(len(df.columns) * 0.8))
    print(f"  Dropped {before - len(df)} rows (duplicates + high-NaN)")
    return df.reset_index(drop=True)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Replace inf/-inf before median — inf poisons median computation
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.difference([LABEL_COL, ATTACK_COL])
    df[cat_cols] = df[cat_cols].fillna("unknown")
    return df


def encode_labels(df: pd.DataFrame):
    """Returns df with integer binary label + attack string label."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"Column '{LABEL_COL}' not found. Available: {list(df.columns)}")
    y_binary = df[LABEL_COL].astype(int)
    y_attack = df[ATTACK_COL] if ATTACK_COL in df.columns else None
    return y_binary, y_attack


def select_features(df: pd.DataFrame, extra_cols: list = None) -> pd.DataFrame:
    """Keep only known informative features that exist in this dataset."""
    wanted = [c for c in TARGET_FEATURES if c in df.columns]
    if extra_cols:
        wanted += [c for c in extra_cols if c in df.columns and c not in wanted]
    missing = [c for c in TARGET_FEATURES if c not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} target features not in dataset — {missing[:5]}...")
    print(f"  Using {len(wanted)} features")
    return df[wanted]


def remove_low_variance(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    kept = X.columns[sel.get_support()].tolist()
    dropped = [c for c in X.columns if c not in kept]
    if dropped:
        print(f"  Variance threshold dropped: {dropped}")
    return X[kept]


def remove_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"  Correlation drop ({threshold}): {to_drop}")
    return X.drop(columns=to_drop)


def build_scaler_pipeline():
    return Pipeline([("scaler", RobustScaler())])


def run_full_pipeline(path: Path = DATASET_PATH, sample_n: int = None):
    df = load_raw(path)
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=RANDOM_STATE)
        print(f"  Sampled to {len(df)} rows")

    df = drop_junk(df)
    df = fill_missing(df)

    y_binary, y_attack = encode_labels(df)
    X = select_features(df)
    X = remove_low_variance(X)
    X = remove_correlated(X)

    feature_names = X.columns.tolist()
    print(f"  Final feature set ({len(feature_names)}): {feature_names}")

    X_arr = X.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_binary.values,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_binary.values
    )

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, DATA_PROCESSED / "scaler.pkl")
    joblib.dump(feature_names, DATA_PROCESSED / "feature_names.pkl")

    np.save(DATA_PROCESSED / "X_train.npy", X_train)
    np.save(DATA_PROCESSED / "X_test.npy", X_test)
    np.save(DATA_PROCESSED / "y_train.npy", y_train)
    np.save(DATA_PROCESSED / "y_test.npy", y_test)

    if y_attack is not None:
        y_attack_train, y_attack_test = train_test_split(
            y_attack.values, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y_binary.values
        )
        np.save(DATA_PROCESSED / "y_attack_train.npy", y_attack_train)
        np.save(DATA_PROCESSED / "y_attack_test.npy", y_attack_test)

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Saved preprocessed data → {DATA_PROCESSED}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run_full_pipeline()
