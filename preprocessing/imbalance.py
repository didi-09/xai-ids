import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                k_neighbors: int = 5, random_state: int = 42):
    """SMOTE oversampling — only ever called on training data."""
    before = Counter(y_train)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after = Counter(y_res)
    print(f"  SMOTE: {dict(before)} → {dict(after)}")
    return X_res, y_res


def apply_hybrid_resample(X_train: np.ndarray, y_train: np.ndarray,
                          random_state: int = 42):
    """SMOTE minority up + random undersample majority — good for extreme imbalance."""
    pipeline = ImbPipeline([
        ("over", SMOTE(random_state=random_state)),
        ("under", RandomUnderSampler(random_state=random_state)),
    ])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    print(f"  Hybrid resample: {Counter(y_res)}")
    return X_res, y_res
