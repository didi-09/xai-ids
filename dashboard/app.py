"""
XAI-IDS Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
import sys
import time
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR
from explainability.shap_engine import SHAPEngine
from dashboard.components.stats_panel import render_stats
from dashboard.components.prediction_feed import render_feed
from dashboard.components.shap_panel import render_shap_explanation, render_global_importance

st.set_page_config(
    page_title="XAI-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ XAI-IDS")
    st.caption("Explainable Intrusion Detection System for IoT Networks")
    st.divider()

    mode = st.radio("Mode", ["Dataset Simulation", "Upload CSV"])
    speed = st.slider("Simulation speed (flows/sec)", 1, 50, 10)
    top_n_shap = st.slider("SHAP top features", 3, 10, 5)

    st.divider()
    if st.button("Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Load model + SHAP ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_engine():
    model_path = MODELS_DIR / "rf_binary.pkl"
    if not model_path.exists():
        return None
    return SHAPEngine(model_path=model_path)


@st.cache_data(show_spinner="Loading test data...")
def load_test_data():
    path = DATA_PROCESSED / "X_test.npy"
    if not path.exists():
        return None
    return np.load(path)


@st.cache_data(show_spinner="Computing global SHAP...")
def load_global_shap(max_samples=300):
    engine = load_engine()
    X_test = load_test_data()
    if engine is None or X_test is None:
        return None, None
    sv = engine.compute_shap(X_test, max_samples=max_samples)
    return sv, X_test[:max_samples]


engine = load_engine()
X_test = load_test_data()

# ── Session state ─────────────────────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = 0
if "selected_pred" not in st.session_state:
    st.session_state.selected_pred = None

# ── No model warning ──────────────────────────────────────────────────────────
if engine is None:
    st.error("No trained model found. Run `python models/train_binary.py` first.")
    st.stop()

if X_test is None:
    st.warning("No preprocessed data found. Run `python preprocessing/pipeline.py` first.")
    st.stop()

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🛡️ XAI Intrusion Detection System")
st.caption("Real-time explainable IoT network traffic classification")

tab_live, tab_explain, tab_global = st.tabs(["Live Feed", "Explanation", "Global SHAP"])

# ── Tab 1: Live Feed ──────────────────────────────────────────────────────────
with tab_live:
    col_stats, col_feed = st.columns([1, 2])

    with col_stats:
        render_stats(st.session_state.predictions)

    with col_feed:
        st.subheader("Prediction Feed")
        render_feed(st.session_state.predictions)

    run_col, stop_col = st.columns(2)
    run_sim = run_col.button("▶ Run Simulation Step", use_container_width=True)
    auto_run = stop_col.checkbox("Auto-run (continuous)")

    if run_sim or auto_run:
        idx = st.session_state.sim_idx
        if idx < len(X_test):
            batch = X_test[idx:idx + speed]
            for row in batch:
                result = engine.explain_single(row, top_n=top_n_shap)
                result["timestamp"] = datetime.now().strftime("%H:%M:%S")
                st.session_state.predictions.append(result)
                st.session_state.selected_pred = result
            st.session_state.sim_idx += len(batch)
        else:
            st.info("Simulation complete — all test flows processed.")

        if auto_run:
            time.sleep(1.0 / max(speed, 1))
            st.rerun()
        else:
            st.rerun()

# ── Tab 2: Explanation ────────────────────────────────────────────────────────
with tab_explain:
    pred = st.session_state.selected_pred
    if pred is None:
        st.info("Run the simulation to see per-prediction explanations.")
    else:
        render_shap_explanation(pred)

        st.divider()
        st.subheader("SHAP Force Plot")
        try:
            # Find the raw feature vector for this prediction
            # (simplified: use most recent sim index - 1)
            last_idx = max(0, st.session_state.sim_idx - 1)
            html = engine.waterfall_html(X_test[last_idx])
            import streamlit.components.v1 as components
            components.html(html, height=200, scrolling=False)
        except Exception as e:
            st.warning(f"Force plot unavailable: {e}")

# ── Tab 3: Global SHAP ────────────────────────────────────────────────────────
with tab_global:
    st.subheader("Global Feature Importance")
    sv, X_sub = load_global_shap()
    if sv is not None:
        render_global_importance(sv, X_sub, engine.feature_names)
    else:
        st.warning("Global SHAP not available — model or data missing.")
