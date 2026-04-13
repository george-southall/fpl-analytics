"""Page 2 — Fixture Difficulty Calendar.

Heat map: teams × upcoming GWs, coloured by difficulty (1=easy, 5=hard).
Toggle between attacking view (xG for) and defensive view (CS probability).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import (
    DIFFICULTY_COLOURS,
    load_fixture_difficulty,
    load_fpl_data,
    upcoming_gw_list,
)

st.set_page_config(page_title="Fixture Difficulty · FPL Analytics", layout="wide")
st.title("📅 Fixture Difficulty Calendar")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    n_gws = st.slider("Gameweeks to show", 1, 8, 6)
    view = st.radio("View", ["Difficulty (1-5)", "xG For", "xG Against", "Clean Sheet %"])

# ── Load data ─────────────────────────────────────────────────────────────────
gws = upcoming_gw_list(n_gws)
if not gws:
    st.warning("No upcoming fixtures found.")
    st.stop()

with st.spinner("Computing fixture difficulty…"):
    try:
        diff_df = load_fixture_difficulty(tuple(gws))
    except Exception as e:
        st.error(f"Could not compute fixture difficulty: {e}")
        st.stop()

if diff_df.empty:
    st.warning("No fixture data available for those gameweeks.")
    st.stop()

# ── Pivot the selected metric ─────────────────────────────────────────────────
metric_map = {
    "Difficulty (1-5)": "difficulty",
    "xG For": "xg_for",
    "xG Against": "xg_against",
    "Clean Sheet %": "cs_prob",
}
metric_col = metric_map[view]

pivot = diff_df.pivot_table(
    index="team", columns="gw", values=metric_col, aggfunc="min"
)
pivot = pivot[[gw for gw in gws if gw in pivot.columns]]
teams = sorted(pivot.index.tolist())
pivot = pivot.loc[teams]
col_labels = [f"GW{gw}" for gw in pivot.columns]

z = pivot.values.tolist()
text = []
for row_i, row in enumerate(pivot.itertuples(index=False)):
    text_row = []
    for val in row:
        if np.isnan(val):
            text_row.append("BGW")
        elif metric_col == "difficulty":
            text_row.append(str(int(val)))
        elif metric_col == "cs_prob":
            text_row.append(f"{val*100:.0f}%")
        else:
            text_row.append(f"{val:.2f}")
    text.append(text_row)

# Colour scale: for difficulty, green=easy(1), red=hard(5); reverse for CS%
if metric_col in ("difficulty", "xg_against"):
    colorscale = "RdYlGn_r"
else:
    colorscale = "RdYlGn"

fig = go.Figure(go.Heatmap(
    z=z,
    x=col_labels,
    y=teams,
    colorscale=colorscale,
    text=text,
    texttemplate="%{text}",
    textfont={"size": 13, "color": "black"},
    hovertemplate="Team: %{y}<br>%{x}<br>Value: %{z:.2f}<extra></extra>",
    showscale=True,
    zmin=1 if metric_col == "difficulty" else None,
    zmax=5 if metric_col == "difficulty" else None,
))

fig.update_layout(
    title=f"Fixture {view} — next {n_gws} GWs",
    xaxis=dict(side="top"),
    yaxis=dict(autorange="reversed"),
    template="plotly_white",
    height=max(400, 25 * len(teams)),
    margin=dict(l=120),
)

st.plotly_chart(fig, use_container_width=True)

# ── Key ───────────────────────────────────────────────────────────────────────
if metric_col == "difficulty":
    st.markdown(
        "**Difficulty key:** "
        + "  ".join(
            f'<span style="background:{c};padding:2px 8px;border-radius:4px;color:black">{d}</span>'
            for d, c in DIFFICULTY_COLOURS.items()
        ),
        unsafe_allow_html=True,
    )
    st.caption("BGW = blank gameweek (team has no fixture)")

# ── Detail table ──────────────────────────────────────────────────────────────
with st.expander("Show detailed fixture data"):
    show_cols = ["team", "opponent", "venue", "gw", "xg_for", "xg_against", "cs_prob", "difficulty", "win_prob"]
    st.dataframe(
        diff_df[[c for c in show_cols if c in diff_df.columns]]
        .sort_values(["team", "gw"])
        .style.format({
            "xg_for": "{:.2f}", "xg_against": "{:.2f}",
            "cs_prob": "{:.1%}", "win_prob": "{:.1%}",
        }),
        use_container_width=True,
    )
