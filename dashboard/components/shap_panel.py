import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import shap
import numpy as np


def render_shap_explanation(explanation: dict):
    st.subheader("Prediction Explanation")

    label_color = "red" if explanation["label"] == 1 else "green"
    st.markdown(
        f"**Prediction:** <span style='color:{label_color}; font-size:1.2em'>"
        f"{explanation['label_name']}</span> — "
        f"**Confidence:** {explanation['confidence']:.1%}",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.text(explanation["reasons"])


def render_global_importance(shap_values: np.ndarray, X: np.ndarray, feature_names: list):
    st.subheader("Global Feature Importance (SHAP)")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names,
        plot_type="bar", show=False, ax=ax
    )
    st.pyplot(fig, use_container_width=True)
    plt.close()


def render_force_plot(html_str: str):
    st.subheader("SHAP Force Plot")
    components.html(html_str, height=180, scrolling=False)
