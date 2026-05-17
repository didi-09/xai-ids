import streamlit as st
import pandas as pd
from datetime import datetime


def render_feed(predictions: list[dict], max_rows: int = 50):
    if not predictions:
        st.info("Waiting for traffic...")
        return

    rows = []
    for p in predictions[-max_rows:][::-1]:
        label = p["label_name"]
        conf = p["confidence"]
        ts = p.get("timestamp", "—")
        top_reason = p["reasons"].split("\n")[2] if "\n" in p["reasons"] else "—"
        rows.append({
            "Time": ts,
            "Label": f"🔴 {label}" if label == "Malicious" else f"🟢 {label}",
            "Confidence": f"{conf:.1%}",
            "Top Signal": top_reason.strip(),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
