# XAI-IDS — Explainable Lightweight Intrusion Detection System for IoT Networks

> Random Forest + SHAP + Streamlit — trained on NF-UNSW-NB15-v3 (2.36M flows)

## Results

| Model | F1 | AUC | FNR | Throughput |
|---|---|---|---|---|
| RF (full, N=100, d=15) | 99.94% | 1.000 | 0.02% | 5,519 flows/sec |
| RF (edge, N=25, d=8) | 99.91% | 1.000 | 0.01% | **11,055 flows/sec** |
| Multiclass RF | macro-F1: 63.65% | — | — | — |

## Quick Start

```bash
pip install -r deployment/requirements.txt

# 1. Download NF-UNSW-NB15-v3.csv and place in data/raw/
# 2. Preprocess
python preprocessing/pipeline.py

# 3. Train
python models/train_binary.py
python models/train_binary.py --edge
python models/train_multiclass.py

# 4. Launch dashboard
streamlit run dashboard/app.py

# 5. Launch API
uvicorn api.main:app --port 8000
```

## Project Structure

```
xai-ids/
├── data/               # raw + processed datasets (not committed)
├── models/             # training scripts + saved weights (weights not committed)
├── notebooks/          # EDA script
├── preprocessing/      # cleaning, feature selection, scaling
├── explainability/     # SHAP engine + NLG
├── dashboard/          # Streamlit app + components
├── realtime/           # dataset replay simulator
├── api/                # FastAPI REST endpoint
├── utils/              # config, metrics
├── deployment/         # Dockerfile, requirements.txt
└── report/             # LaTeX paper (main.tex + references.bib)
```

## API

```bash
POST /predict
{"features": [0.1, 0.2, ...]}   # 32 features

POST /predict/batch
{"flows": [[...], [...]]}

GET /model/info
GET /health
```

## Dataset

[NF-UNSW-NB15-v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) — Sarhan et al. 2021


Compile with:
```bash
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
