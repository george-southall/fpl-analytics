"""Page 1 — Player Projections.

Filterable, sortable table showing GW-by-GW expected points for all players.
Click a player to see their full scoring breakdown.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import (
    POSITION_COLOURS,
    load_projections,
    upcoming_gw_list,
)

st.set_page_config(page_title="Player Projections · FPL Analytics", layout="wide")
st.title("📊 Player Projections")

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    horizon = st.slider("GW horizon", 1, 6, 6)
    positions = st.multiselect(
        "Position", ["GK", "DEF", "MID", "FWD"], default=["GK", "DEF", "MID", "FWD"]
    )
    max_price = st.slider("Max price (£m)", 4.0, 16.0, 16.0, step=0.5)
    min_ownership = st.slider("Min ownership (%)", 0.0, 100.0, 0.0, step=0.5)
    min_xmins = st.slider("Min xMins", 0, 90, 0, step=10)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading projections…"):
    try:
        df = load_projections(horizon=horizon)
    except Exception as e:
        st.error(f"Could not load projections: {e}")
        st.stop()

# Identify GW columns
gw_cols = [c for c in df.columns if c.startswith("GW") and c.endswith("_xPts")]

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    df["position"].isin(positions)
    & (df["price"] <= max_price)
    & (df["selected_by_percent"] >= min_ownership)
    & (df["xMins"] >= min_xmins)
)
filtered = df[mask].copy()

st.caption(f"Showing {len(filtered)} players · GW horizon: {horizon}")

# ── Summary metrics ───────────────────────────────────────────────────────────
top = filtered.nlargest(1, "total_xPts")
if not top.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players shown", len(filtered))
    c2.metric("Top projected", top.iloc[0].get("web_name", top.iloc[0].get("name", "")),
              f"{top.iloc[0]['total_xPts']:.1f} xPts")
    c3.metric("Avg xPts", f"{filtered['total_xPts'].mean():.1f}")
    c4.metric("Avg price", f"£{filtered['price'].mean():.1f}m")

# ── Main table ────────────────────────────────────────────────────────────────
display_cols = ["name", "team", "position", "price", "selected_by_percent", "xMins"] + gw_cols + ["total_xPts"]
display_df = filtered[display_cols].copy()
display_df = display_df.rename(columns={"selected_by_percent": "own%", "total_xPts": "Total xPts"})

# Colour gradient on GW and total columns
gw_display = [c for c in display_df.columns if c.startswith("GW")]
pts_cols = gw_display + ["Total xPts"]

styled = (
    display_df.style
    .background_gradient(subset=pts_cols, cmap="RdYlGn", axis=None)
    .format({c: "{:.1f}" for c in pts_cols})
    .format({"price": "£{:.1f}m", "own%": "{:.1f}%", "xMins": "{:.0f}"})
)

st.dataframe(styled, use_container_width=True, height=500)

# ── Export ────────────────────────────────────────────────────────────────────
csv = filtered[display_cols].to_csv(index=False)
st.download_button("⬇ Export CSV", csv, "fpl_projections.csv", "text/csv")

# ── Player detail ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Player breakdown")

player_names = filtered.sort_values("total_xPts", ascending=False)["name"].tolist() if "name" in filtered.columns else []
if not player_names:
    player_names = filtered.sort_values("total_xPts", ascending=False).get("web_name", pd.Series()).tolist()

selected_name = st.selectbox("Select a player", player_names)

if selected_name:
    row = filtered[filtered.get("name", filtered.get("web_name", pd.Series())) == selected_name]
    if row.empty:
        row = filtered[filtered.get("web_name", pd.Series(dtype=str)) == selected_name]

    if not row.empty:
        p = row.iloc[0]
        pos = p["position"]
        colour = POSITION_COLOURS.get(pos, "#888")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Position", pos)
        c2.metric("Team", p["team"])
        c3.metric("Price", f"£{p['price']:.1f}m")
        c4.metric("Ownership", f"{p['selected_by_percent']:.1f}%")
        c5.metric("Total xPts", f"{p['total_xPts']:.1f}")

        # GW-by-GW bar chart
        if gw_cols:
            gw_labels = [c.replace("_xPts", "") for c in gw_cols]
            gw_values = [float(p.get(c, 0)) for c in gw_cols]

            fig = go.Figure(go.Bar(
                x=gw_labels, y=gw_values,
                marker_color=colour,
                text=[f"{v:.1f}" for v in gw_values],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{selected_name} — GW-by-GW Projected Points",
                xaxis_title="Gameweek",
                yaxis_title="xPts",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
