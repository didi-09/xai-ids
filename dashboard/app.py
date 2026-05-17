"""
XAI-IDS Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_PROCESSED, MODELS_DIR
from explainability.shap_engine import SHAPEngine
from dashboard.components.stats_panel import render_metrics
from dashboard.components.prediction_feed import render_feed, render_explanation
from dashboard.components.shap_panel import render_global_importance, render_beeswarm

st.set_page_config(
    page_title="XAI-IDS — Network Threat Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model...")
def load_engine():
    model_path = MODELS_DIR / "rf_binary.pkl"
    if not model_path.exists():
        return None
    return SHAPEngine(model_path=model_path)


@st.cache_resource(show_spinner="Loading network data...")
def load_test_data():
    path = DATA_PROCESSED / "X_test.npy"
    if not path.exists():
        return None
    return np.load(path)


@st.cache_resource(show_spinner="Analyzing feature importance (one-time)...")
def load_global_shap(max_samples=300):
    _engine = load_engine()
    _X = load_test_data()
    if _engine is None or _X is None:
        return None, None
    sv = _engine.compute_shap(_X, max_samples=max_samples)
    return sv, _X[:max_samples]


engine = load_engine()
X_test = load_test_data()

# ── Guard: missing model or data ──────────────────────────────────────────────
if engine is None:
    st.error("⚠️ No trained model found. Run `python models/train_binary.py` first.")
    st.stop()
if X_test is None:
    st.error("⚠️ No processed data found. Run `python preprocessing/pipeline.py` first.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = 0
if "selected_pred" not in st.session_state:
    st.session_state.selected_pred = None
if "running" not in st.session_state:
    st.session_state.running = False

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("## 🛡️ Network Threat Monitor")
    st.caption("Every decision is explained in plain English. No ML knowledge required.")
with col_status:
    analyzed = st.session_state.sim_idx
    total_available = len(X_test)
    progress = analyzed / total_available if total_available > 0 else 0
    st.markdown(f"**Progress:** {analyzed:,} / {total_available:,} flows")
    st.progress(progress)

st.divider()

# ── Metric cards ──────────────────────────────────────────────────────────────
render_metrics(st.session_state.predictions)

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 1, 3])

with ctrl1:
    batch_size = st.select_slider(
        "Flows per step",
        options=[1, 5, 10, 25, 50],
        value=10,
        label_visibility="collapsed",
        help="How many network flows to analyze each time you click Analyze"
    )
    st.caption(f"Analyzing **{batch_size}** flows per step")

with ctrl2:
    run_btn = st.button("▶ Analyze Next Batch", use_container_width=True, type="primary")

with ctrl3:
    reset_btn = st.button("🔄 Reset", use_container_width=True)

with ctrl4:
    auto_run = st.checkbox("▶▶ Auto-run continuously", value=False,
                           help="Keeps analyzing flows automatically until all are processed")

if reset_btn:
    st.session_state.predictions = []
    st.session_state.sim_idx = 0
    st.session_state.selected_pred = None
    st.rerun()

# ── Run simulation ─────────────────────────────────────────────────────────────
if run_btn or auto_run:
    idx = st.session_state.sim_idx
    if idx >= len(X_test):
        st.success(f"✅ All {len(X_test):,} flows analyzed.")
    else:
        batch = X_test[idx:idx + batch_size]
        for row in batch:
            result = engine.explain_single(row, top_n=5)
            result["timestamp"] = datetime.now().strftime("%H:%M:%S")
            st.session_state.predictions.append(result)
            st.session_state.selected_pred = result
        st.session_state.sim_idx += len(batch)

        if auto_run:
            time.sleep(0.3)
            st.rerun()
        else:
            st.rerun()

st.divider()

# ── Main split: Feed + Explanation ────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    render_feed(st.session_state.predictions)

with right_col:
    st.markdown("#### Why did the model decide this?")
    pred = st.session_state.selected_pred
    if pred is None:
        st.info("🔍 Click **▶ Analyze Next Batch** to start.\n\nThe explanation for each decision will appear here.")
    else:
        render_explanation(pred)

st.divider()

# ── Advanced section ──────────────────────────────────────────────────────────
with st.expander("📊 Advanced Analysis — Feature Importance (for technical users)"):
    st.markdown(
        "These charts show **which network features matter most** across all analyzed flows. "
        "They answer: *'What patterns does the model look for when detecting threats?'*"
    )
    sv, X_sub = load_global_shap()
    if sv is not None:
        tab_bar, tab_bee = st.tabs(["Feature Importance", "Detailed Impact Chart"])
        with tab_bar:
            render_global_importance(sv, X_sub, engine.feature_names)
        with tab_bee:
            render_beeswarm(sv, X_sub, engine.feature_names)
    else:
        st.warning("Feature importance data not available.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("XAI-IDS · Random Forest + SHAP · NF-UNSW-NB15-v3 · Binary F1: 99.94% · AUC: 1.00 · FNR: 0.02%")
