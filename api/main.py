"""
FastAPI backend for XAI-IDS
Endpoints:
  POST /predict          — classify a single flow
  POST /predict/batch    — classify multiple flows
  GET  /model/info       — model metadata
  GET  /health           — health check

Run: uvicorn api.main:app --reload --port 8000
"""
import sys
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR
from explainability.shap_engine import SHAPEngine

app = FastAPI(title="XAI-IDS API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_engine: Optional[SHAPEngine] = None
_feature_names: Optional[list] = None


def get_engine() -> SHAPEngine:
    global _engine
    if _engine is None:
        model_path = MODELS_DIR / "rf_binary.pkl"
        if not model_path.exists():
            raise HTTPException(503, "Model not trained yet")
        _engine = SHAPEngine(model_path=model_path)
    return _engine


def get_feature_names() -> list:
    global _feature_names
    if _feature_names is None:
        path = MODELS_DIR / "feature_names.pkl"
        if not path.exists():
            raise HTTPException(503, "Feature names not available")
        _feature_names = joblib.load(path)
    return _feature_names


class FlowInput(BaseModel):
    features: list[float]


class BatchInput(BaseModel):
    flows: list[list[float]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    names = get_feature_names()
    return {
        "model": "RandomForest (binary)",
        "n_features": len(names),
        "feature_names": names,
    }


@app.post("/predict")
def predict(flow: FlowInput):
    engine = get_engine()
    names = get_feature_names()
    if len(flow.features) != len(names):
        raise HTTPException(422, f"Expected {len(names)} features, got {len(flow.features)}")
    x = np.array(flow.features, dtype=np.float32)
    result = engine.explain_single(x)
    return result


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    engine = get_engine()
    names = get_feature_names()
    results = []
    for features in batch.flows:
        if len(features) != len(names):
            results.append({"error": f"Expected {len(names)} features"})
            continue
        x = np.array(features, dtype=np.float32)
        results.append(engine.explain_single(x))
    return {"results": results, "count": len(results)}
