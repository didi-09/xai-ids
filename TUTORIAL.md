# XAI-IDS — Complete Tutorial

This guide walks you through every step: environment setup, dataset preparation,
training, evaluation, running the dashboard, and using the API.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Dataset Setup](#3-dataset-setup)
4. [Phase 1 — Exploratory Data Analysis](#4-phase-1--exploratory-data-analysis)
5. [Phase 2 — Preprocessing](#5-phase-2--preprocessing)
6. [Phase 3 — Training the Models](#6-phase-3--training-the-models)
7. [Phase 4 — Evaluating the Models](#7-phase-4--evaluating-the-models)
8. [Phase 5 — SHAP Explainability](#8-phase-5--shap-explainability)
9. [Phase 6 — Streamlit Dashboard](#9-phase-6--streamlit-dashboard)
10. [Phase 7 — FastAPI REST Endpoint](#10-phase-7--fastapi-rest-endpoint)
11. [Phase 8 — Real-Time Simulation](#11-phase-8--real-time-simulation)
12. [Phase 9 — Edge Benchmark](#12-phase-9--edge-benchmark)
13. [Generating Report Figures](#13-generating-report-figures)
14. [Compiling the LaTeX Report](#14-compiling-the-latex-report)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| pip | 23+ | `pip install --upgrade pip` |
| RAM | 8 GB minimum | 16 GB recommended for full dataset |
| Disk | 3 GB free | dataset ~800 MB, processed data ~1.5 GB |
| LaTeX | TeX Live 2022+ | Only needed for report compilation |

---

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/didi-09/xai-ids.git
cd xai-ids

# Install all Python dependencies
pip install -r deployment/requirements.txt
```

**Verify the installation:**
```bash
python3 -c "import sklearn, shap, streamlit, fastapi; print('All imports OK')"
```

Expected output:
```
All imports OK
```

---

## 3. Dataset Setup

### Download NF-UNSW-NB15-v3

1. Go to: https://staff.itee.uq.edu.au/marius/NIDS_datasets/
2. Download **NF-UNSW-NB15-v3.csv**
3. Place it in the `data/raw/` directory:

```bash
mkdir -p data/raw
mv ~/Downloads/NF-UNSW-NB15-v3.csv data/raw/
```

### Tell the system where the dataset is

The dataset path is configured via an environment variable.
Set it once in your terminal session:

```bash
# If you placed it in data/raw/ (default — no env var needed):
# The system will look for data/raw/NF-UNSW-NB15-v3.csv automatically

# If your dataset is somewhere else, set this:
export XAI_IDS_DATASET="/path/to/your/NF-UNSW-NB15-v3.csv"
```

**Tip:** Add the export to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### Verify the dataset loads

```bash
python3 -c "
from utils.config import DATASET_PATH
import pandas as pd
df = pd.read_csv(DATASET_PATH, nrows=5)
print('Dataset OK:', df.shape[1], 'columns found')
print('First columns:', list(df.columns[:5]))
"
```

Expected output:
```
Dataset OK: 55 columns found
First columns: ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR']
```

---

## 4. Phase 1 — Exploratory Data Analysis

Run the EDA script to understand the dataset structure:

```bash
python3 notebooks/01_eda.py
```

**What it does:**
- Prints dataset shape, column types, missing values, duplicate count
- Shows binary label distribution (Benign vs Malicious)
- Shows attack type breakdown (Exploits, DoS, Reconnaissance, etc.)
- Saves a 10,000-row sample to `data/samples/sample_10k.csv`
- Generates two plots in `data/samples/`:
  - `eda_class_dist.png` — bar chart of class imbalance
  - `eda_correlation.png` — feature correlation heatmap

**Expected output (key numbers):**
```
Raw shape: (2365424, 55)
Duplicate rows: 14815
Label distribution:
  0 (Benign):    2237731  (94.6%)
  1 (Malicious):  127693  (5.4%)
Attack type distribution:
  Benign:        2237731
  Exploits:        42748
  Fuzzers:         33816
  ...
```

**Runtime:** ~2 minutes (reads 2.36M rows)

---

## 5. Phase 2 — Preprocessing

This is the most important step. Run it once and the results are cached.

```bash
python3 preprocessing/pipeline.py
```

**What it does (in order):**
1. Loads the full dataset (2.36M rows)
2. Removes 14,815 duplicate rows
3. Replaces infinity values (from zero-duration flow division)
4. Fills missing values with column medians
5. Drops IP addresses, timestamps, and protocol-specific sparse columns
6. Selects 43 behavioral features
7. Drops 11 highly correlated features (|r| > 0.95)
8. Produces a final set of **32 features**
9. Applies `RobustScaler` (fit on train, transform both)
10. Saves everything to `data/processed/`:
    - `X_train.npy` — 1,880,487 × 32 training features
    - `X_test.npy`  — 470,122 × 32 test features
    - `y_train.npy` — binary labels (train)
    - `y_test.npy`  — binary labels (test)
    - `y_attack_train.npy` — attack type strings (train)
    - `y_attack_test.npy`  — attack type strings (test)
    - `scaler.pkl`  — fitted RobustScaler
    - `feature_names.pkl` — list of 32 feature names

**Expected output:**
```
Loading dataset from .../NF-UNSW-NB15-v3.csv ...
  Raw shape: (2365424, 55)
  Dropped 14815 rows (duplicates + high-NaN)
  Using 43 features
  Correlation drop (0.95): ['OUT_PKTS', 'CLIENT_TCP_FLAGS', ...]
  Final feature set (32): ['L4_SRC_PORT', 'L4_DST_PORT', ...]
  Train: (1880487, 32) | Test: (470122, 32)
  Saved preprocessed data → .../data/processed
```

**Runtime:** ~3–4 minutes

> **Note:** You only need to run this once. All subsequent steps load from `data/processed/`.

---

## 6. Phase 3 — Training the Models

### 6a. Binary Classifier (Benign vs Malicious)

```bash
# Full model (100 trees, depth 15) — best accuracy
python3 models/train_binary.py

# Edge-optimised model (25 trees, depth 8) — fastest inference
python3 models/train_binary.py --edge

# Without SMOTE (not needed — class_weight=balanced handles imbalance)
python3 models/train_binary.py --no-smote
```

**Expected output (full model):**
```
=== Binary RF Training ===
  Features: 32 | Train: 1880487 | Test: 470122
  Training...
  Accuracy : 0.9999
  F1       : 0.9994
  Precision: 0.9989
  Recall   : 0.9998
  ROC-AUC  : 1.0000
  FNR      : 0.0002
  Model saved → .../models/rf_binary.pkl
```

**Runtime:** ~5–8 minutes (full), ~2 minutes (edge)

### 6b. Multiclass Classifier (per attack type)

```bash
# Random Forest multiclass
python3 models/train_multiclass.py --model rf

# XGBoost multiclass (alternative)
python3 models/train_multiclass.py --model xgb
```

**Expected output:**
```
=== Multiclass Training (RF) ===
  Training...
  Accuracy : 0.9843
  F1       : 0.6365   ← macro-F1 (low due to rare classes like Worms)
  Weighted F1 is ~99.2% (dominated by Benign class)
  Model saved → .../models/rf_multiclass.pkl
```

**Runtime:** ~8–12 minutes

### Saved model files

| File | Description |
|---|---|
| `models/rf_binary.pkl` | Full binary classifier |
| `models/rf_binary_edge.pkl` | Lightweight edge classifier |
| `models/rf_multiclass.pkl` | Multiclass RF (10 attack types) |
| `models/xgb_multiclass.pkl` | Multiclass XGBoost |
| `models/label_encoder_multiclass.pkl` | Attack label encoder |
| `models/feature_names.pkl` | List of 32 feature names |
| `models/metrics_binary.json` | Full evaluation metrics (binary) |
| `models/metrics_binary_edge.json` | Edge model metrics |
| `models/metrics_multiclass_rf.json` | Multiclass metrics |

---

## 7. Phase 4 — Evaluating the Models

View the saved metrics directly:

```bash
# Binary model metrics
python3 -c "
import json
with open('models/metrics_binary.json') as f:
    m = json.load(f)
print(f'Accuracy : {m[\"accuracy\"]:.4f}')
print(f'F1       : {m[\"f1\"]:.4f}')
print(f'Precision: {m[\"precision\"]:.4f}')
print(f'Recall   : {m[\"recall\"]:.4f}')
print(f'ROC-AUC  : {m[\"roc_auc\"]:.4f}')
print(f'FNR      : {m[\"false_negative_rate\"]:.4f}')
"
```

**Compare full vs edge:**
```bash
python3 -c "
import json
for name in ['metrics_binary.json', 'metrics_binary_edge.json']:
    with open(f'models/{name}') as f:
        m = json.load(f)
    label = 'Full' if 'edge' not in name else 'Edge'
    print(f'{label}: F1={m[\"f1\"]:.4f}  FNR={m[\"false_negative_rate\"]:.4f}  AUC={m[\"roc_auc\"]:.4f}')
"
```

Expected:
```
Full: F1=0.9994  FNR=0.0002  AUC=1.0000
Edge: F1=0.9991  FNR=0.0001  AUC=1.0000
```

---

## 8. Phase 5 — SHAP Explainability

### Generate global SHAP plots

```bash
python3 explainability/shap_engine.py
```

**What it produces:**
- `models/shap_global.png` — bar chart of top features by mean |SHAP|
- `models/shap_beeswarm.png` — beeswarm plot showing impact direction

**Runtime:** ~30 seconds (200 samples)

### Test per-prediction explanation in Python

```bash
python3 -c "
import numpy as np
from explainability.shap_engine import SHAPEngine

engine = SHAPEngine()
X_test = np.load('data/processed/X_test.npy')

# Explain a single flow (index 27 is a malicious sample)
result = engine.explain_single(X_test[27])
print(f'Prediction : {result[\"label_name\"]}')
print(f'Confidence : {result[\"confidence\"]:.1%}')
print()
print(result['reasons'])
"
```

**Expected output:**
```
Prediction : Malicious
Confidence : 100.0%

Traffic classified as: Malicious

Key reasons:
  - High MIN_TTL → more malicious (impact: 0.2540)
  - High MIN_IP_PKT_LEN → more malicious (impact: 0.0962)
```

---

## 9. Phase 6 — Streamlit Dashboard

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at: **http://localhost:8501**

### Dashboard tabs

**Tab 1 — Live Feed**
- Shows real-time prediction results as you step through the dataset
- Click **"▶ Run Simulation Step"** to classify one batch of flows
- Enable **"Auto-run"** checkbox for continuous streaming
- Left panel shows: total flows, attack count, benign count, attack type pie chart
- Right panel shows: scrollable prediction feed with label, confidence, top SHAP reason

**Tab 2 — Explanation**
- Click any entry in the feed to see its full explanation
- Shows: predicted label, confidence percentage
- Shows: full ranked list of contributing features with direction and SHAP impact
- Shows: SHAP force plot (interactive HTML)

**Tab 3 — Global SHAP**
- Shows the global feature importance bar chart across 300 test samples
- Answers: "Which network features matter most for detecting attacks?"

### Sidebar controls

| Control | Description |
|---|---|
| Mode | Dataset Simulation or Upload CSV |
| Simulation speed | 1–50 flows per step |
| SHAP top features | How many features to show in explanations |
| Reset Session | Clear all predictions and start over |

### Test with your own CSV

1. Sidebar → Mode → **Upload CSV**
2. Upload a CSV with the same 32 feature columns (see `models/feature_names.pkl`)
3. The system will classify each row and show explanations

```bash
# Get the required feature names
python3 -c "import joblib; print(joblib.load('models/feature_names.pkl'))"
```

---

## 10. Phase 7 — FastAPI REST Endpoint

### Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs** (Swagger UI)

### Endpoints

#### Health check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

#### Model info
```bash
curl http://localhost:8000/model/info
```
```json
{
  "model": "RandomForest (binary)",
  "n_features": 32,
  "feature_names": ["L4_SRC_PORT", "L4_DST_PORT", ...]
}
```

#### Classify a single flow
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [443, 52341, 6, 91, 1500, 800, 10, 40.5, 18, 250, 0, 1000, 64, 1500, 200, 300, 100, 5, 0, 0, 0, 0, 65535, 8192, 0.5, 2.1, 1.2, 0.8, 0.3, 1.8, 1.1, 0.6]}'
```
```json
{
  "label": 0,
  "label_name": "Benign",
  "confidence": 0.99,
  "shap_values": [...],
  "feature_names": [...],
  "reasons": "Traffic classified as: Benign\n\nKey reasons:\n  - Low MIN_TTL → more benign (impact: 0.2149)"
}
```

> **Note:** The 32 feature values must match this exact order:
> `L4_SRC_PORT, L4_DST_PORT, PROTOCOL, L7_PROTO, IN_BYTES, OUT_BYTES, IN_PKTS, TCP_FLAGS, FLOW_DURATION_MILLISECONDS, DURATION_OUT, MIN_TTL, LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES, SRC_TO_DST_AVG_THROUGHPUT, NUM_PKTS_UP_TO_128_BYTES, NUM_PKTS_128_TO_256_BYTES, NUM_PKTS_256_TO_512_BYTES, NUM_PKTS_512_TO_1024_BYTES, NUM_PKTS_1024_TO_1514_BYTES, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT, SRC_TO_DST_IAT_MIN, SRC_TO_DST_IAT_MAX, SRC_TO_DST_IAT_AVG, SRC_TO_DST_IAT_STDDEV, DST_TO_SRC_IAT_MIN, DST_TO_SRC_IAT_MAX, DST_TO_SRC_IAT_AVG, DST_TO_SRC_IAT_STDDEV`

#### Classify a batch of flows
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"flows": [[443,52341,6,...], [80,43210,6,...]]}'
```
```json
{
  "results": [{"label": 0, ...}, {"label": 1, ...}],
  "count": 2
}
```

#### Test with a real flow from the dataset
```bash
python3 -c "
import numpy as np, json, requests
X = np.load('data/processed/X_test.npy')
# Send flow index 27 (known malicious)
r = requests.post('http://localhost:8000/predict',
    json={'features': X[27].tolist()})
d = r.json()
print(d['label_name'], f'{d[\"confidence\"]:.1%}')
print(d['reasons'])
"
```

---

## 11. Phase 8 — Real-Time Simulation

### Run the simulator

```bash
# Simulate 100 flows at 50 flows/sec (default)
python3 realtime/simulator.py

# Custom: 500 flows at 10 flows/sec, quiet mode
python3 realtime/simulator.py --n 500 --delay 0.1 --quiet

# Fast test: 20 flows, print all
python3 realtime/simulator.py --n 20 --delay 0.01
```

**Expected output:**
```
Simulating 30 flows at 50.0 flows/sec...
------------------------------------------------------------
[01:16:26.670] ✓ Benign     conf=1.00 | - Low MIN_TTL → more benign (impact: 0.2149)
[01:16:26.708] ✓ Benign     conf=1.00 | - Low MIN_TTL → more benign (impact: 0.2290)
[01:16:26.746] ✓ Malicious  conf=1.00 | - High MIN_TTL → more malicious (impact: 0.2540)
...
------------------------------------------------------------
Accuracy on 30 flows: 100.00%
```

**Legend:**
- `✓` = correct prediction (matches ground truth)
- `✗` = wrong prediction
- `conf=1.00` = 100% model confidence
- The reason line shows the single most influential feature

---

## 12. Phase 9 — Edge Benchmark

```bash
python3 models/edge_benchmark.py
```

**Expected output:**
```
rf_binary.pkl
  Latency      : 0.1812 ms/sample
  Throughput   : 5,519 flows/sec
  Peak RAM     : 0.27 MB
  Model file   : 1.69 MB

rf_binary_edge.pkl
  Latency      : 0.0905 ms/sample
  Throughput   : 11,055 flows/sec
  Peak RAM     : 0.15 MB
  Model file   : 0.25 MB
```

**IoT gateway targets (Raspberry Pi 4 class):**

| Target | Requirement | Full Model | Edge Model |
|---|---|---|---|
| Latency | < 5 ms | 0.18 ms ✓ | 0.09 ms ✓ |
| Throughput | > 500 flows/sec | 5,519 ✓ | 11,055 ✓ |
| Model size | < 50 MB | 1.69 MB ✓ | 0.25 MB ✓ |

Both models exceed all IoT deployment targets by a wide margin.

---

## 13. Generating Report Figures

```bash
python3 models/generate_plots.py
```

Generates 9 figures in `report/figures/`:

| Figure | Content |
|---|---|
| `eda_class_dist.png` | Binary + attack type class distribution |
| `eda_correlation.png` | Feature correlation heatmap |
| `confusion_matrix.png` | TP/TN/FP/FN for binary RF |
| `roc_curve.png` | ROC curves (full vs edge, both AUC=1.00) |
| `multiclass_f1.png` | Per-class F1 bar chart (colour-coded) |
| `feature_importance.png` | RF built-in importance (top 15) |
| `shap_global.png` | SHAP mean feature importance bar |
| `shap_beeswarm.png` | SHAP beeswarm (direction + magnitude) |
| `throughput_tradeoff.png` | Speed vs accuracy scatter |

---

## 14. Compiling the LaTeX Report

```bash
cd report

# Full compile sequence (required for citations to resolve)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `report/main.pdf` — 20 pages

**Requirements:**
```bash
# Ubuntu/Kali/Debian
sudo apt install texlive-full

# Verify
pdflatex --version
bibtex --version
```

---

## 15. Troubleshooting

### `FileNotFoundError: data/processed/X_test.npy`
You haven't run preprocessing yet. Run:
```bash
python3 preprocessing/pipeline.py
```

### `FileNotFoundError: models/rf_binary.pkl`
You haven't trained the model yet. Run:
```bash
python3 models/train_binary.py
```

### `ModuleNotFoundError: No module named 'shap'`
```bash
pip install -r deployment/requirements.txt
```

### Dataset path not found
```bash
export XAI_IDS_DATASET="/full/path/to/NF-UNSW-NB15-v3.csv"
```

### Streamlit port already in use
```bash
streamlit run dashboard/app.py --server.port 8502
```

### API port already in use
```bash
uvicorn api.main:app --port 8001
```

### SHAP takes too long
SHAP on large samples is slow. The engine is capped at 500 samples by default.
Reduce further if needed:
```python
engine.compute_shap(X_test, max_samples=100)
```

### Out of memory during training
Train on a sample instead of the full dataset:
```python
# In pipeline.py — add sample_n argument
from preprocessing.pipeline import run_full_pipeline
run_full_pipeline(sample_n=200000)
```

---

## Quick Reference — Run Order

```bash
# One-time setup
pip install -r deployment/requirements.txt
export XAI_IDS_DATASET="/path/to/NF-UNSW-NB15-v3.csv"

# Data pipeline
python3 notebooks/01_eda.py           # explore dataset
python3 preprocessing/pipeline.py     # preprocess + save

# Training
python3 models/train_binary.py        # full binary model
python3 models/train_binary.py --edge # edge binary model
python3 models/train_multiclass.py    # multiclass RF

# Evaluation
python3 models/edge_benchmark.py      # latency + throughput
python3 explainability/shap_engine.py # SHAP plots
python3 models/generate_plots.py      # all report figures

# Real-time
python3 realtime/simulator.py         # terminal simulation

# Services (run in separate terminals)
streamlit run dashboard/app.py        # dashboard → :8501
uvicorn api.main:app --port 8000      # API → :8000

# Report
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
