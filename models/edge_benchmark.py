"""Edge deployment benchmark — measures latency, RAM, throughput."""
import sys, time, tracemalloc, json
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR

def benchmark(model_path: Path, n_samples: int = 1000):
    model = joblib.load(model_path)
    X_test = np.load(DATA_PROCESSED / "X_test.npy")
    X_bench = X_test[:n_samples]

    # Warm-up
    _ = model.predict(X_bench[:10])

    tracemalloc.start()
    t0 = time.perf_counter()
    preds = model.predict(X_bench)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = t1 - t0
    latency_ms = (elapsed / n_samples) * 1000
    throughput = n_samples / elapsed
    peak_mb = peak / 1024**2
    model_size_mb = model_path.stat().st_size / 1024**2

    result = {
        "model": model_path.name,
        "n_samples": n_samples,
        "latency_ms_per_sample": round(latency_ms, 4),
        "throughput_flows_per_sec": round(throughput, 1),
        "peak_ram_mb": round(peak_mb, 2),
        "model_file_mb": round(model_size_mb, 2),
    }
    return result

if __name__ == "__main__":
    results = []
    for name in ["rf_binary.pkl", "rf_binary_edge.pkl"]:
        path = MODELS_DIR / name
        if path.exists():
            r = benchmark(path)
            results.append(r)
            print(f"\n{r['model']}")
            print(f"  Latency      : {r['latency_ms_per_sample']:.4f} ms/sample")
            print(f"  Throughput   : {r['throughput_flows_per_sec']:,.0f} flows/sec")
            print(f"  Peak RAM     : {r['peak_ram_mb']:.2f} MB")
            print(f"  Model file   : {r['model_file_mb']:.2f} MB")

    out = MODELS_DIR / "edge_benchmark.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")
