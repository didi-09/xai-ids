import streamlit as st


def render_metrics(predictions: list):
    total = len(predictions)
    threats = sum(1 for p in predictions if p["label"] == 1)
    safe = total - threats
    rate = (threats / total * 100) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows Analyzed", f"{total:,}")
    c2.metric("🔴 Threats", f"{threats:,}")
    c3.metric("🟢 Safe", f"{safe:,}")
    c4.metric("Threat Rate", f"{rate:.1f}%", delta=None)
