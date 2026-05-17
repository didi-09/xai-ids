# Explainable AI IDS — Project Plan
**"Explainable Intrusion Detection System for IoT Networks"**

---

## Dataset Decision

| Dataset | Size | Attack Types | Notes |
|---|---|---|---|
| **NF-UNSW-NB15-v3** | ~2.5M flows | 9 attack categories | NetFlow features, cleaner labels, broader attack diversity |
| **UNSW_2018_IoT_Botnet_Full5pc_3** | ~5% sample of full | Botnet-focused | IoT-specific, but narrower scope |

**Recommendation: Start with NF-UNSW-NB15-v3**
- More balanced attack type variety for multiclass extension
- NetFlow features map directly to real IoT gateway capture (tshark/Zeek output)
- "v3" has corrected labels vs earlier versions
- If you want to extend to botnet-specific later, add UNSW_2018 as a second dataset

---

## Architecture (Final Vision)

```
Raw PCAP / Network Flow
        │
        ▼
┌─────────────────────┐
│  Feature Extraction  │  ← CICFlowMeter / tshark / Zeek
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Preprocessing      │  ← pandas: drop NaN, encode, normalize
│   + Feature Select   │  ← top 20 features via correlation + SHAP
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Random Forest      │  ← Phase 3 baseline (binary)
│   → XGBoost          │  ← Phase 5 multiclass upgrade
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  SHAP Explainer      │  ← Global (feature importance)
│  + LIME (optional)   │  ← Local (per-prediction reasoning)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Streamlit Dashboard │  ← Predictions + SHAP plots + alerts
└─────────────────────┘
```

---

## Phase 1 — Dataset Understanding (Week 1)

**Deliverable:** Jupyter notebook `01_eda.ipynb`

Tasks:
- Load dataset, print `.info()`, `.describe()`, `.value_counts()` on label column
- Count NaN per column, duplicates, class imbalance ratio
- Plot: class distribution bar chart, correlation heatmap, feature distributions per class

**Expected findings for NF-UNSW-NB15-v3:**
- ~45 raw features (NetFlow stats: bytes, packets, duration, ports, TCP flags, IAT)
- Labels: Benign, DoS, DDoS, Reconnaissance, Exploits, Backdoor, Analysis, Fuzzers, Worms, Generic
- Expect significant class imbalance (Benign >> attack classes)

**Key columns to identify:**
```
IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT,
PROTOCOL, L7_PROTO, IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS,
TCP_FLAGS, FLOW_DURATION_MILLISECONDS, Label, Attack
```

---

## Phase 2 — Preprocessing (Week 2)

**Deliverable:** `preprocessing/pipeline.py` + notebook `02_preprocessing.ipynb`

### Step 1: Clean
```python
df.drop_duplicates(inplace=True)
df.dropna(thresh=len(df.columns)*0.8, inplace=True)   # drop rows missing >20% features
df.fillna(df.median(numeric_only=True), inplace=True)  # fill remaining
```

### Step 2: Encode
```python
# Categorical: Protocol, L7_PROTO → LabelEncoder or frequency encoding
# IP addresses: drop or convert to /24 subnet integer (reduces cardinality)
```

### Step 3: Normalize
```python
from sklearn.preprocessing import RobustScaler  # better than MinMax for outlier-heavy traffic data
```

### Step 4: Feature Selection
Strategy:
1. Drop near-zero variance columns (`VarianceThreshold`)
2. Drop highly correlated pairs (Pearson > 0.95, keep the one with higher label correlation)
3. After baseline model: use SHAP to validate and prune further

**Target: 15–25 features**

### Step 5: Handle Imbalance
```python
from imblearn.over_sampling import SMOTE
# Apply SMOTE on training set only — never on test set
```

---

## Phase 3 — Baseline Model (Week 3–4)

**Deliverable:** `models/rf_binary.pkl` + notebook `03_baseline.ipynb`

### Binary First (Benign vs. Malicious)
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,          # cap depth for edge deployment
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```

### Evaluation
```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
```

Report must include:
- Precision, Recall, F1 per class
- ROC-AUC
- **Confusion matrix** — false negatives are critical (missed attacks)

**Target metrics (binary):** F1 ≥ 0.95, FNR < 3%

### After Binary Success → Multiclass (Week 5)
Same RF but with all attack labels. Report macro + weighted F1.

---

## Phase 4 — Explainability (Week 5–6)

**Deliverable:** `explainability/shap_engine.py` + notebook `04_explainability.ipynb`

### Global Explainability
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global bar plot — top features overall
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Beeswarm — feature impact distribution
shap.summary_plot(shap_values, X_test)
```

### Local Explainability (per prediction)
```python
# Force plot — why THIS flow was classified as attack
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])

# Waterfall — cleaner single-prediction view
shap.waterfall_plot(shap.Explanation(
    values=shap_values[i],
    base_values=explainer.expected_value,
    data=X_test.iloc[i]
))
```

### Human-Readable Reasoning Engine
```python
def explain_prediction(flow_row, shap_vals, feature_names, threshold=0.1):
    """Convert SHAP values to natural language."""
    pairs = sorted(zip(shap_vals, feature_names), reverse=True)
    reasons = []
    for val, feat in pairs:
        if abs(val) >= threshold:
            direction = "high" if val > 0 else "low"
            reasons.append(f"- {direction.capitalize()} {feat} (impact: {val:+.3f})")
    return "\n".join(reasons[:5])
```

**Example output:**
```
Traffic classified as: DDoS [confidence: 97.3%]

Reasons:
- High IN_PKTS: unusually high packet count pushed toward malicious (+0.42)
- High FLOW_DURATION_MILLISECONDS: extremely short flow duration (+0.38)
- High TCP_FLAGS: SYN flood pattern detected (+0.31)
- Low OUT_BYTES: asymmetric traffic ratio (+0.19)
- Suspicious L4_DST_PORT: known DDoS target port (+0.15)
```

---

## Phase 5 — Dashboard (Week 6)

**Deliverable:** `dashboard/app.py`

### Streamlit Layout
```
┌─────────────────────────────────────────────────┐
│  XAI-IDS Dashboard                              │
├──────────────┬──────────────────────────────────┤
│ Stats Panel  │  Live Prediction Feed             │
│ Total flows  │  [timestamp] [src→dst] [label]    │
│ Attack count │  [confidence] [top reason]        │
│ Benign count │                                   │
├──────────────┼──────────────────────────────────┤
│ Attack Type  │  SHAP Explanation (selected flow) │
│ Distribution │  Global Feature Importance        │
│ (Pie/Bar)    │  Waterfall / Force Plot           │
└──────────────┴──────────────────────────────────┘
```

### Core Components
```python
import streamlit as st
import shap
import streamlit.components.v1 as components

# Sidebar: file upload or live simulation toggle
# Main: st.dataframe for predictions, plotly for charts
# SHAP: render HTML force plot via components.html()
```

---

## Phase 6 — Real-Time Simulation (Week 7)

**Deliverable:** `realtime/simulator.py`

### Stage 1: Dataset Replay
```python
import time, pandas as pd

def simulate_stream(df, model, explainer, delay=0.05):
    for _, row in df.iterrows():
        features = preprocess_row(row)
        pred = model.predict([features])[0]
        shap_vals = explainer.shap_values([features])[pred]
        explanation = explain_prediction(features, shap_vals)
        yield {"prediction": pred, "explanation": explanation, "raw": row}
        time.sleep(delay)
```

### Stage 2: Live Capture (Later)
```
tshark -i eth0 -T fields -e ... | CICFlowMeter → preprocessor → model
```

---

## Phase 7 — Edge Optimization (Week 8)

**Deliverable:** `models/rf_edge.pkl` + benchmark report

### Optimization Techniques
| Technique | Method | Expected Saving |
|---|---|---|
| Fewer trees | n_estimators=25 (vs 100) | 4× faster inference |
| Max depth cap | max_depth=8 | 30% memory reduction |
| Feature reduction | top 15 SHAP features only | 40% preprocessing speedup |
| Quantization | sklearn → ONNX → quantize | 2–4× memory reduction |

### Benchmark Script
```python
import time, tracemalloc, psutil

tracemalloc.start()
t0 = time.perf_counter()
pred = model.predict(X_test[:1000])
t1 = time.perf_counter()
mem = tracemalloc.get_peak()[1] / 1024**2

print(f"Latency: {(t1-t0)/1000*1000:.2f}ms/sample")
print(f"Peak RAM: {mem:.1f}MB")
print(f"Throughput: {1000/(t1-t0):.0f} flows/sec")
```

**Target for edge (Raspberry Pi 4 class):**
- Latency < 5ms/flow
- RAM < 50MB model footprint
- Throughput > 500 flows/sec

---

## Phase 8 — Enhancements (Week 9–10)

Priority order:

**1. Multiclass → already planned in Phase 3**

**2. Hybrid IDS (High Research Value)**
```
Snort/Suricata signatures → known attack fast path
ML model → unknown / zero-day detection
Fusion layer → combine alerts
```

**3. Autonomous Response Hook (connects to SOARGanism)**
```python
if pred == "DDoS" and confidence > 0.95:
    block_ip(src_ip)
    notify_soar(event)
    deploy_honeypot_if_configured()
```

**4. Federated Learning (Paper-worthy)**
- Each IoT node trains locally on its traffic
- Only gradient updates shared (not raw flows = privacy preserved)
- Aggregate at edge gateway

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| ML | scikit-learn | 1.4+ |
| Boost | XGBoost | 2.0+ |
| Explainability | shap | 0.45+ |
| Imbalance | imbalanced-learn | 0.12+ |
| Dashboard | Streamlit | 1.35+ |
| API | FastAPI | 0.111+ |
| Data | pandas, numpy | latest |
| Notebooks | Jupyter Lab | 4.x |
| Real-time | Scapy / tshark | - |
| Deployment | Docker | 24+ |
| ONNX | onnxruntime | 1.18+ |

---

## Folder Structure

```
xai-ids/
│
├── data/
│   ├── raw/                  # original dataset files
│   ├── processed/            # cleaned, encoded, scaled
│   └── samples/              # small subsets for testing
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_edge_benchmark.ipynb
│
├── preprocessing/
│   ├── pipeline.py           # full sklearn Pipeline
│   ├── feature_select.py
│   └── imbalance.py
│
├── models/
│   ├── rf_binary.pkl
│   ├── rf_multiclass.pkl
│   ├── xgb_multiclass.pkl
│   └── rf_edge.onnx          # quantized for edge
│
├── explainability/
│   ├── shap_engine.py        # TreeExplainer wrapper
│   ├── lime_engine.py        # optional
│   └── nlg.py               # natural language generation from SHAP
│
├── dashboard/
│   ├── app.py                # Streamlit main app
│   ├── components/
│   │   ├── prediction_feed.py
│   │   ├── shap_panel.py
│   │   └── stats_panel.py
│   └── assets/
│
├── realtime/
│   ├── simulator.py          # dataset replay
│   ├── capture.py            # tshark live capture
│   └── preprocessor.py      # online feature extraction
│
├── api/
│   ├── main.py               # FastAPI endpoints
│   └── schemas.py
│
├── utils/
│   ├── metrics.py
│   ├── logging.py
│   └── config.py
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── PLAN.md                   # this file
└── README.md
```

---

## Timeline

| Week | Phase | Milestone |
|---|---|---|
| 1 | EDA | Dataset fully understood, class distribution mapped |
| 2 | Preprocessing | Clean pipeline, 15–25 features selected |
| 3 | Baseline | Binary RF trained, F1 ≥ 0.95 |
| 4 | Evaluation | Full metrics, confusion matrix, error analysis |
| 5 | Explainability | SHAP global + local working, NLG output |
| 6 | Dashboard | Streamlit app with SHAP integration live |
| 7 | Real-time | Dataset simulation streaming into dashboard |
| 8 | Optimization | Edge-optimized model benchmarked |
| 9 | Multiclass + Hybrid | All attack types classified, Snort integration |
| 10 | Paper + Demo | Written up, demo video recorded |

---

## Research Paper Outline

**Title:** "Explainable Lightweight Intrusion Detection System for IoT Networks Using Ensemble Learning"

**Abstract:** IDS + XAI + edge constraints = novel contribution

**Sections:**
1. Introduction — IoT threat landscape, black-box problem
2. Related Work — survey of ML-IDS and XAI-IDS papers
3. Dataset & Preprocessing — NF-UNSW-NB15-v3 analysis
4. Methodology — RF + SHAP pipeline
5. Evaluation — metrics, comparison vs. prior work
6. Explainability Analysis — case studies (DDoS, Recon, Botnet)
7. Edge Deployment — latency/RAM benchmarks
8. Conclusion + Future Work (Federated Learning)

**Target venues:** IEEE Access, Computers & Security, MDPI Sensors

