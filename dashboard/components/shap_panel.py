import streamlit as st
import matplotlib.pyplot as plt
import shap
import numpy as np


def render_global_importance(shap_values: np.ndarray, X: np.ndarray, feature_names: list):
    from dashboard.components.prediction_feed import FEATURE_LABELS
    readable_names = [FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f in feature_names]

    shap.summary_plot(shap_values, X, feature_names=readable_names,
                      plot_type="bar", show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    st.pyplot(fig, use_container_width=True)
    plt.close("all")
    st.caption("Bar length = average impact on the model's decision across all analyzed flows. "
               "Longer bar = feature mattered more.")


def render_beeswarm(shap_values: np.ndarray, X: np.ndarray, feature_names: list):
    from dashboard.components.prediction_feed import FEATURE_LABELS
    readable_names = [FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f in feature_names]

    shap.summary_plot(shap_values, X, feature_names=readable_names, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    st.pyplot(fig, use_container_width=True)
    plt.close("all")
    st.caption("Each dot = one network flow. Red = high feature value, Blue = low. "
               "Position on X-axis = how much it pushed the decision toward Malicious (right) or Benign (left).")
