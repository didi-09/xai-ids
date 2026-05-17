"""
Dataset replay simulator — feeds rows from the processed test set
through the model and SHAP engine as if they were live traffic.

Usage:
  python realtime/simulator.py --delay 0.05 --n 200
"""
import sys
import time
import argparse
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR
from explainability.shap_engine import SHAPEngine


def simulate(delay: float = 0.1, n: int = 100, verbose: bool = True):
    engine = SHAPEngine()
    X_test = np.load(DATA_PROCESSED / "X_test.npy")
    y_test = np.load(DATA_PROCESSED / "y_test.npy")

    print(f"Simulating {n} flows at {1/delay:.1f} flows/sec...")
    print("-" * 60)

    results = []
    for i in range(min(n, len(X_test))):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        row = X_test[i]
        result = engine.explain_single(row)
        result["timestamp"] = ts
        result["true_label"] = int(y_test[i])
        results.append(result)

        if verbose:
            status = "✓" if result["label"] == result["true_label"] else "✗"
            print(
                f"[{ts}] {status} {result['label_name']:10s} "
                f"conf={result['confidence']:.2f} | "
    f"{next((l.strip() for l in result['reasons'].split(chr(10)) if l.strip().startswith('-')), '')}"
            )

        time.sleep(delay)

    correct = sum(1 for r in results if r["label"] == r["true_label"])
    print("-" * 60)
    print(f"Accuracy on {n} flows: {correct/n:.2%}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    simulate(delay=args.delay, n=args.n, verbose=not args.quiet)
