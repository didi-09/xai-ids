import streamlit as st


# Human-readable labels for raw feature names
FEATURE_LABELS = {
    "IN_PKTS":                    "Inbound packet count",
    "OUT_BYTES":                  "Outbound data volume",
    "IN_BYTES":                   "Inbound data volume",
    "FLOW_DURATION_MILLISECONDS": "Flow duration",
    "MIN_TTL":                    "Time-to-live (TTL) value",
    "TCP_FLAGS":                  "TCP flag pattern",
    "SRC_TO_DST_IAT_MIN":         "Minimum inter-packet gap",
    "SRC_TO_DST_IAT_AVG":         "Average inter-packet gap",
    "SRC_TO_DST_IAT_STDDEV":      "Inter-packet timing variability",
    "DST_TO_SRC_IAT_MIN":         "Response inter-packet gap",
    "DST_TO_SRC_IAT_AVG":         "Average response gap",
    "LONGEST_FLOW_PKT":           "Largest packet size",
    "SHORTEST_FLOW_PKT":          "Smallest packet size",
    "MIN_IP_PKT_LEN":             "Minimum IP packet length",
    "SRC_TO_DST_SECOND_BYTES":    "Bytes per second (outbound)",
    "DST_TO_SRC_SECOND_BYTES":    "Bytes per second (inbound)",
    "SRC_TO_DST_AVG_THROUGHPUT":  "Average outbound throughput",
    "L4_SRC_PORT":                "Source port",
    "L4_DST_PORT":                "Destination port",
    "PROTOCOL":                   "Network protocol",
    "L7_PROTO":                   "Application protocol",
    "TCP_WIN_MAX_IN":             "TCP receive window (in)",
    "TCP_WIN_MAX_OUT":            "TCP receive window (out)",
    "NUM_PKTS_UP_TO_128_BYTES":   "Small packet count (≤128B)",
    "NUM_PKTS_128_TO_256_BYTES":  "Medium packet count (128–256B)",
    "NUM_PKTS_256_TO_512_BYTES":  "Medium packet count (256–512B)",
    "NUM_PKTS_512_TO_1024_BYTES": "Large packet count (512–1024B)",
    "NUM_PKTS_1024_TO_1514_BYTES":"Max-size packet count",
    "DURATION_OUT":               "Outbound flow duration",
}

ATTACK_CONTEXT = {
    "IN_PKTS":                    "High packet counts are typical of DDoS floods and port scans.",
    "FLOW_DURATION_MILLISECONDS": "Attack tools operate much faster or slower than human browsing.",
    "MIN_TTL":                    "Crafted attack packets often have unusual TTL values.",
    "TCP_FLAGS":                  "Malformed TCP flag combinations signal SYN floods or RST injections.",
    "SRC_TO_DST_IAT_MIN":         "Near-zero inter-packet gaps indicate automated attack tooling.",
    "SRC_TO_DST_IAT_AVG":         "Highly regular timing is characteristic of scripted attacks.",
    "MIN_IP_PKT_LEN":             "Unusually small packets are common in reconnaissance probes.",
    "LONGEST_FLOW_PKT":           "Abnormal packet sizes can indicate data exfiltration or fuzzing.",
}


def humanize_reason(raw_reason: str) -> list[str]:
    """Convert raw SHAP reason lines into plain-English bullets."""
    bullets = []
    for line in raw_reason.split("\n"):
        line = line.strip()
        if not line.startswith("-"):
            continue
        # e.g. "- High MIN_TTL → more malicious (impact: 0.254)"
        try:
            body = line.lstrip("- ").strip()
            direction_feat, _ = body.split("→")
            direction, feat_raw = direction_feat.strip().split(" ", 1)
            feat_raw = feat_raw.strip()
            feat_label = FEATURE_LABELS.get(feat_raw, feat_raw.replace("_", " ").title())
            context = ATTACK_CONTEXT.get(feat_raw, "")
            direction_word = "abnormally high" if direction == "High" else "abnormally low"
            text = f"**{feat_label}** is {direction_word}."
            if context:
                text += f" {context}"
            bullets.append(text)
        except Exception:
            bullets.append(line.lstrip("- ").strip())
    return bullets


def render_feed(predictions: list, max_rows: int = 60):
    if not predictions:
        st.info("No traffic analyzed yet. Click **Analyze** to start.")
        return

    st.markdown("#### Recent Traffic")
    for p in predictions[-max_rows:][::-1]:
        is_threat = p["label"] == 1
        bg = "#fff5f5" if is_threat else "#f0fff4"
        border = "#fc8181" if is_threat else "#68d391"
        icon = "🔴" if is_threat else "🟢"
        status = "THREAT" if is_threat else "Safe"
        conf = p["confidence"]
        ts = p.get("timestamp", "")
        top_reason = humanize_reason(p["reasons"])
        top_line = top_reason[0] if top_reason else "—"

        with st.container():
            st.markdown(
                f"""<div style="border-left:4px solid {border}; background:{bg};
                    padding:8px 12px; border-radius:4px; margin-bottom:6px; color:#1a202c;">
                    <span style="font-size:1.05em; font-weight:700; color:{border};">{icon} {status}</span>
                    &nbsp;&nbsp;<span style="color:#4a5568; font-size:0.88em;">{ts} &nbsp;·&nbsp; Confidence: <b>{conf:.0%}</b></span><br>
                    <span style="font-size:0.9em; color:#2d3748;">{top_line}</span>
                </div>""",
                unsafe_allow_html=True,
            )


def render_explanation(pred: dict):
    """Full plain-English explanation card for a selected prediction."""
    is_threat = pred["label"] == 1
    icon = "⚠️" if is_threat else "✅"
    verdict = "THREAT DETECTED" if is_threat else "SAFE TRAFFIC"
    color = "#c53030" if is_threat else "#276749"
    bg = "#fff5f5" if is_threat else "#f0fff4"
    border = "#fc8181" if is_threat else "#68d391"
    conf = pred["confidence"]

    st.markdown(
        f"""<div style="border:2px solid {border}; background:{bg};
            padding:16px 20px; border-radius:8px; margin-bottom:12px; color:#1a202c;">
            <div style="font-size:1.3em; font-weight:700; color:{color};">{icon} {verdict}</div>
            <div style="margin-top:4px; color:#4a5568;">Model confidence: <b style="color:#1a202c;">{conf:.1%}</b></div>
        </div>""",
        unsafe_allow_html=True,
    )

    bullets = humanize_reason(pred["reasons"])
    if bullets:
        if is_threat:
            st.markdown("**Why was this flow flagged as a threat?**")
        else:
            st.markdown("**Why was this flow classified as safe?**")

        for i, b in enumerate(bullets, 1):
            st.markdown(f"{i}. {b}")
    else:
        st.markdown("_No significant signals found for this flow._")

    with st.expander("🔬 Technical details (SHAP values)"):
        raw = pred.get("reasons", "")
        st.code(raw, language=None)
        st.caption("SHAP (SHapley Additive exPlanations) values show how much each "
                   "network feature pushed the model toward its decision. "
                   "Positive = pushed toward Malicious. Negative = pushed toward Benign.")
