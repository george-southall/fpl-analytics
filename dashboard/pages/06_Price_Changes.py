"""Page 6 — Price Change Alerts.

XGBoost classifier predicts rise / hold / fall for every player based on
current-GW net transfer rates, form, ownership, and lagged price momentum.
Heuristic alerts are shown when the model confidence is low.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px
import streamlit as st

from dashboard.data_loader import load_fpl_data, load_price_alerts, load_price_model

st.set_page_config(page_title="Price Change Alerts · FPL Analytics", layout="wide")
st.title("💰 Price Change Alerts")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    positions = st.multiselect(
        "Position", ["GK", "DEF", "MID", "FWD"], default=["GK", "DEF", "MID", "FWD"]
    )
    min_ownership = st.slider("Min ownership (%)", 0.0, 100.0, 1.0)
    min_confidence = st.slider("Min model confidence", 0.0, 1.0, 0.0, step=0.05)
    show_holds = st.checkbox("Show hold predictions", value=False)
    retrain_btn = st.button("🔄 Retrain model", help="Force model retraining from fresh data")

# ── Load data ─────────────────────────────────────────────────────────────────
if retrain_btn:
    st.cache_resource.clear()
    st.rerun()

with st.spinner("Loading price alerts…"):
    try:
        alerts = load_price_alerts()
        model = load_price_model()
    except Exception as e:
        st.error(f"Could not load price alerts: {e}")
        st.stop()

if alerts.empty:
    st.warning("No alert data available. Check FPL API connectivity.")
    st.stop()

# ── Model info ────────────────────────────────────────────────────────────────
if model.is_trained and model.precision_ is not None:
    st.success(
        f"Model trained · validation macro precision: **{model.precision_:.2f}**",
        icon="✅",
    )
else:
    st.info(
        "Model not trained — showing heuristic alerts only. "
        "Click **Retrain model** in the sidebar to train on current-season data.",
        icon="🚧",
    )

st.divider()

# ── Filter ────────────────────────────────────────────────────────────────────
df = alerts.copy()
if "position" in df.columns:
    df = df[df["position"].isin(positions)]
df = df[df["selected_by_percent"].astype(float) >= min_ownership]
df = df[df["confidence"] >= min_confidence]
if not show_holds:
    df = df[df["alert"] != "hold"]

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rise candidates", len(df[df["alert"] == "rise"]))
c2.metric("Fall candidates", len(df[df["alert"] == "fall"]))
c3.metric("Hold", len(df[df["alert"] == "hold"]))
c4.metric("Players analysed", len(alerts))

st.divider()

# ── Alert tables ──────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

DISPLAY_COLS = ["web_name", "team_name", "position", "price", "selected_by_percent",
                "net_transfers_event", "net_transfer_rate_event",
                "prob_rise", "prob_fall", "confidence"]
RENAME = {
    "web_name": "Player", "team_name": "Team", "selected_by_percent": "Own%",
    "net_transfers_event": "Net Transfers", "net_transfer_rate_event": "Rate",
    "prob_rise": "P(rise)", "prob_fall": "P(fall)",
}
display_cols = [c for c in DISPLAY_COLS if c in df.columns]
fmt = {
    "price": "£{:.1f}m", "Own%": "{:.1f}%",
    "Net Transfers": "{:,.0f}", "Rate": "{:.4f}",
    "P(rise)": "{:.2f}", "P(fall)": "{:.2f}", "confidence": "{:.2f}",
}

with col_l:
    st.subheader("🟢 Rise candidates")
    rise = (
        df[df["alert"] == "rise"]
        .sort_values("prob_rise" if "prob_rise" in df.columns else "net_transfer_rate_event",
                     ascending=False)
        [display_cols].head(20)
        .rename(columns=RENAME)
    )
    if rise.empty:
        st.caption("No rise candidates with current filters.")
    else:
        st.dataframe(rise.style.format(fmt, na_rep="—"), use_container_width=True, height=420)

with col_r:
    st.subheader("🔴 Fall candidates")
    fall = (
        df[df["alert"] == "fall"]
        .sort_values("prob_fall" if "prob_fall" in df.columns else "net_transfer_rate_event",
                     ascending=False)
        [display_cols].head(20)
        .rename(columns=RENAME)
    )
    if fall.empty:
        st.caption("No fall candidates with current filters.")
    else:
        st.dataframe(fall.style.format(fmt, na_rep="—"), use_container_width=True, height=420)

st.divider()

# ── Probability scatter ───────────────────────────────────────────────────────
st.subheader("Rise vs Fall probability — all players")

plot_df = alerts.copy()
plot_df = plot_df[plot_df["selected_by_percent"].astype(float) >= min_ownership]
if "position" in plot_df.columns:
    plot_df = plot_df[plot_df["position"].isin(positions)]

if not plot_df.empty and "prob_rise" in plot_df.columns and plot_df["prob_rise"].sum() > 0:
    fig = px.scatter(
        plot_df,
        x="prob_fall",
        y="prob_rise",
        color="position",
        size=plot_df["selected_by_percent"].astype(float).clip(lower=0.5),
        hover_name="web_name",
        hover_data={"price": ":.1f", "confidence": ":.2f", "alert": True},
        labels={"prob_fall": "P(fall)", "prob_rise": "P(rise)"},
        title="Price change model probabilities (bubble size = ownership)",
        color_discrete_map={"GK": "#f6c90e", "DEF": "#00c2a8", "MID": "#6ecbf5", "FWD": "#e84855"},
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.4)
    fig.add_vline(x=0.5, line_dash="dot", line_color="red", opacity=0.4)
    fig.update_layout(template="plotly_white", height=480)
    st.plotly_chart(fig, use_container_width=True)
else:
    # Fall back to net-transfers scatter when model not trained
    if "net_transfers_event" not in plot_df.columns:
        plot_df["net_transfers_event"] = (
            plot_df.get("transfers_in_event", 0) - plot_df.get("transfers_out_event", 0)
        )
    plot_df2 = plot_df[plot_df["net_transfers_event"].abs() > 0]
    if not plot_df2.empty:
        fig = px.scatter(
            plot_df2,
            x="selected_by_percent",
            y="net_transfers_event",
            color="position",
            hover_name="web_name",
            hover_data={"price": ":.1f"},
            labels={"selected_by_percent": "Ownership (%)", "net_transfers_event": "Net transfers"},
            title="Net transfers vs ownership",
            color_discrete_map={"GK": "#f6c90e", "DEF": "#00c2a8", "MID": "#6ecbf5", "FWD": "#e84855"},
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Model: XGBoost classifier · features: net transfer rate, ownership %, price, "
    "form (5-GW rolling), price momentum · training: walk-forward CV on current-season GW histories"
)
