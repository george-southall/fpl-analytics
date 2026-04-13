"""Page 6 — Price Change Alerts.

Displays net transfer data and flags players near price change thresholds.
The full XGBoost price change model is built in Phase 5; this page shows
the heuristic alerts available from FPL API transfer data in the interim.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data_loader import load_fpl_data

st.set_page_config(page_title="Price Change Alerts · FPL Analytics", layout="wide")
st.title("💰 Price Change Alerts")

st.info(
    "**Phase 5 in progress.** The full XGBoost price predictor will power this page. "
    "For now, this page surfaces heuristic alerts based on live net transfer data from the FPL API.",
    icon="🚧",
)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading FPL data…"):
    try:
        players, _, _, _ = load_fpl_data()
    except Exception as e:
        st.error(f"Could not load FPL data: {e}")
        st.stop()

# ── Compute net transfer metrics ──────────────────────────────────────────────
df = players.copy()
df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]
df["net_transfer_pct"] = (df["net_transfers"] / df["transfers_in_event"].sum().clip(min=1)) * 100

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    positions = st.multiselect(
        "Position", ["GK", "DEF", "MID", "FWD"], default=["GK", "DEF", "MID", "FWD"]
    )
    min_ownership = st.slider("Min ownership (%)", 0.0, 100.0, 1.0)
    alert_threshold = st.slider("Alert threshold — net transfer %", 0.01, 1.0, 0.05, step=0.01)

mask = df["position"].isin(positions) & (df["selected_by_percent"] >= min_ownership)
filtered = df[mask].copy()

# ── Summary ───────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
rising = filtered[filtered["net_transfers"] > 0]
falling = filtered[filtered["net_transfers"] < 0]
c1.metric("Players being bought", len(rising))
c2.metric("Players being sold", len(falling))
c3.metric("Net neutral", len(filtered) - len(rising) - len(falling))

st.divider()

# ── Alert tables ──────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🟢 Rise candidates")
    rise_df = (
        filtered[filtered["net_transfer_pct"] >= alert_threshold]
        .sort_values("net_transfer_pct", ascending=False)
        [["web_name", "team_name", "position", "price", "selected_by_percent", "net_transfers", "net_transfer_pct"]]
        .head(20)
        .rename(columns={"web_name": "Player", "team_name": "Team",
                         "selected_by_percent": "Own%", "net_transfer_pct": "Net %"})
    )
    if rise_df.empty:
        st.caption("No players above threshold.")
    else:
        st.dataframe(
            rise_df.style.format({"price": "£{:.1f}m", "Own%": "{:.1f}%",
                                  "Net %": "{:.3f}%", "net_transfers": "{:,.0f}"}),
            use_container_width=True, height=400,
        )

with col_r:
    st.subheader("🔴 Fall candidates")
    fall_df = (
        filtered[filtered["net_transfer_pct"] <= -alert_threshold]
        .sort_values("net_transfer_pct", ascending=True)
        [["web_name", "team_name", "position", "price", "selected_by_percent", "net_transfers", "net_transfer_pct"]]
        .head(20)
        .rename(columns={"web_name": "Player", "team_name": "Team",
                         "selected_by_percent": "Own%", "net_transfer_pct": "Net %"})
    )
    if fall_df.empty:
        st.caption("No players below threshold.")
    else:
        st.dataframe(
            fall_df.style.format({"price": "£{:.1f}m", "Own%": "{:.1f}%",
                                  "Net %": "{:.3f}%", "net_transfers": "{:,.0f}"}),
            use_container_width=True, height=400,
        )

st.divider()

# ── Net transfers scatter ─────────────────────────────────────────────────────
st.subheader("Net transfers vs ownership")
plot_df = filtered[filtered["net_transfers"].abs() > 0].copy()
plot_df["name"] = plot_df["web_name"]

if not plot_df.empty:
    fig = px.scatter(
        plot_df.nlargest(200, "net_transfers").append(plot_df.nsmallest(50, "net_transfers")),
        x="selected_by_percent",
        y="net_transfers",
        color="position",
        size=plot_df.nlargest(200, "net_transfers").append(
            plot_df.nsmallest(50, "net_transfers")
        )["price"].clip(lower=1),
        hover_name="name",
        hover_data={"price": ":.1f", "net_transfers": ":,.0f"},
        labels={"selected_by_percent": "Ownership (%)", "net_transfers": "Net transfers this GW"},
        title="Net Transfers vs Ownership (bubble size = price)",
        color_discrete_map={"GK": "#f6c90e", "DEF": "#00c2a8", "MID": "#6ecbf5", "FWD": "#e84855"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "**Coming in Phase 5:** XGBoost classifier predicting price rise/fall/hold "
    "with precision > 0.65, trained on 3+ seasons of historical FPL data using "
    "walk-forward cross-validation."
)
