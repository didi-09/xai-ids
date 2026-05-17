import streamlit as st
import plotly.express as px
import pandas as pd


def render_stats(predictions: list[dict]):
    if not predictions:
        st.info("No predictions yet.")
        return

    total = len(predictions)
    attacks = sum(1 for p in predictions if p["label"] == 1)
    benign = total - attacks

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", total)
    col2.metric("Attacks", attacks, delta=None)
    col3.metric("Benign", benign)

    if attacks > 0:
        labels = [p.get("attack_type", "Malicious") for p in predictions if p["label"] == 1]
        dist = pd.Series(labels).value_counts().reset_index()
        dist.columns = ["Attack Type", "Count"]
        fig = px.pie(dist, names="Attack Type", values="Count",
                     title="Attack Type Distribution", hole=0.4)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
