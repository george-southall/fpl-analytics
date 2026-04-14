"""Page 4 — Squad Optimiser.

Runs the PuLP LP to select an optimal 15-man squad and displays it
on an interactive pitch graphic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import POSITION_COLOURS, load_fpl_data, load_model, upcoming_gw_list
from fpl_analytics.optimiser.captain_picker import pick_captain
from fpl_analytics.optimiser.squad_optimiser import optimise_squad
from fpl_analytics.projections.fixture_difficulty import compute_fixture_difficulty
from fpl_analytics.projections.projection_engine import run_projections

st.set_page_config(page_title="Squad Optimiser · FPL Analytics", layout="wide")
st.title("🧮 Squad Optimiser")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    budget = st.slider("Budget (£m)", 80.0, 115.0, 110.0, step=0.5)
    horizon = st.radio("Projection window", [1, 3, 6], index=2, horizontal=True)
    run_btn = st.button("⚡ Optimise squad", type="primary", use_container_width=True)

xpts_col = "total_xPts"

if not run_btn:
    st.info("Configure your budget and projection window in the sidebar, then click **Optimise squad**.")
    st.stop()

# ── Run optimisation ─────────────────────────────────────────────────────────
with st.spinner("Running optimiser…"):
    try:
        players, _, fixtures, _ = load_fpl_data()
        ts = load_model()
        projections = run_projections(players, fixtures, ts.model, horizon=horizon)
        solution = optimise_squad(projections, xpts_col=xpts_col, budget=budget)
        gws = upcoming_gw_list(horizon)
        diff_df = compute_fixture_difficulty(fixtures, ts.model, gws)
    except Exception as e:
        st.error(f"Optimisation failed: {e}")
        st.stop()

# Build fixture schedule lookup: team → list of "GW{n}(H/A)" strings
_team_fixtures: dict[str, list[str]] = {}
if not diff_df.empty:
    for _, row in diff_df.iterrows():
        team = row["team"]
        label = f"GW{row['gw']}({row['venue']})"
        if row.get("is_dgw"):
            label += "×2"
        _team_fixtures.setdefault(team, []).append(label)


def _fixture_tag(team_name: str) -> str:
    """Return a compact fixture string for a player's team."""
    fx = _team_fixtures.get(team_name, [])
    return " · ".join(fx) if fx else "BGW"

import pandas as pd
squad = pd.concat([solution.starters, solution.bench]).reset_index(drop=True)
d, m, f = solution.formation

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Formation", f"{d}-{m}-{f}")
c2.metric("Total cost", f"£{solution.total_cost:.1f}m")
c3.metric("Budget remaining", f"£{budget - solution.total_cost:.1f}m")
c4.metric(f"Total xPts ({horizon} GW)", f"{solution.total_xpts:.1f}")

st.divider()

# ── Captain recommendation ────────────────────────────────────────────────────
cap = pick_captain(squad, xpts_col=xpts_col)
st.markdown(
    f"**★ Captain:** {cap.captain} ({cap.captain_xpts:.1f} xPts, {cap.captain_ownership:.1f}% owned)  &nbsp;&nbsp;"
    f"**☆ Vice:** {cap.vice_captain} ({cap.vice_captain_xpts:.1f} xPts)"
    + (f"  &nbsp;&nbsp;**◆ Differential:** {cap.differential_captain} "
       f"({cap.differential_captain_xpts:.1f} xPts, {cap.differential_captain_ownership:.1f}% owned)"
       if cap.differential_captain else ""),
    unsafe_allow_html=True,
)

st.divider()

# ── Pitch graphic ─────────────────────────────────────────────────────────────
st.subheader("Starting XI")

def _pitch_positions(formation: tuple[int, int, int]) -> list[tuple[float, float, str]]:
    """Return (x, y, position_label) for all 11 starting slots."""
    d, m, f = formation
    slots = []
    # GK
    slots.append((0.5, 0.07, "GK"))
    # Defenders
    for i in range(d):
        slots.append(((i + 1) / (d + 1), 0.25, "DEF"))
    # Midfielders
    for i in range(m):
        slots.append(((i + 1) / (m + 1), 0.50, "MID"))
    # Forwards
    for i in range(f):
        slots.append(((i + 1) / (f + 1), 0.75, "FWD"))
    return slots


def _pitch_figure(starters: "pd.DataFrame", formation: tuple, xpts_col: str) -> go.Figure:
    fig = go.Figure()

    # Pitch background
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                  fillcolor="#2d8a4e", line=dict(color="white", width=2))
    # Centre circle
    fig.add_shape(type="circle", x0=0.35, y0=0.42, x1=0.65, y1=0.58,
                  line=dict(color="white", width=1.5))
    # Penalty areas
    for y0, y1 in [(0.0, 0.15), (0.85, 1.0)]:
        fig.add_shape(type="rect", x0=0.25, y0=y0, x1=0.75, y1=y1,
                      line=dict(color="white", width=1.5), fillcolor="rgba(0,0,0,0)")

    slots = _pitch_positions(formation)

    # Sort starters by position for reliable slot assignment
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    sorted_starters = starters.copy()
    sorted_starters["_pos_order"] = sorted_starters["position"].map(pos_order)
    sorted_starters = sorted_starters.sort_values(["_pos_order", xpts_col], ascending=[True, False])

    name_col = "web_name" if "web_name" in sorted_starters.columns else "name"

    for idx, (slot, (_, row)) in enumerate(zip(slots, sorted_starters.iterrows())):
        x, y, pos_label = slot
        colour = POSITION_COLOURS.get(row["position"], "#888")
        name = str(row.get("web_name", row.get("name", "")))
        price = float(row["price"])
        xpts = float(row[xpts_col])

        # Player circle
        fig.add_shape(type="circle",
                      x0=x - 0.055, y0=y - 0.06, x1=x + 0.055, y1=y + 0.06,
                      fillcolor=colour, line=dict(color="white", width=2))

        # Name
        fig.add_annotation(x=x, y=y + 0.01, text=f"<b>{name}</b>",
                            showarrow=False, font=dict(size=9, color="black"),
                            xanchor="center", yanchor="middle")
        # Price + xPts + fixture
        team_name = str(row.get("team", ""))
        fx = _fixture_tag(team_name)
        fig.add_annotation(x=x, y=y - 0.035, text=f"£{price:.1f}m · {xpts:.1f}",
                           showarrow=False, font=dict(size=7.5, color="#222"),
                           xanchor="center", yanchor="middle")
        fig.add_annotation(x=x, y=y - 0.065, text=fx,
                           showarrow=False, font=dict(size=6.5, color="#444"),
                           xanchor="center", yanchor="middle")

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


fig = _pitch_figure(solution.starters, solution.formation, xpts_col)
st.plotly_chart(fig, use_container_width=True)

# ── Bench ─────────────────────────────────────────────────────────────────────
st.subheader("Bench")
bench_cols = st.columns(4)
for i, (_, p) in enumerate(solution.bench.iterrows()):
    name = p.get("web_name", p.get("name", ""))
    with bench_cols[i]:
        fx = _fixture_tag(str(p.get("team", "")))
        st.markdown(
            f"<div style='background:{POSITION_COLOURS.get(p['position'],'#888')};"
            f"padding:8px;border-radius:8px;text-align:center'>"
            f"<b>{name}</b><br>£{p['price']:.1f}m · {p[xpts_col]:.1f} xPts"
            f"<br><span style='font-size:11px;color:#333'>{fx}</span></div>",
            unsafe_allow_html=True,
        )

# ── Full squad table ──────────────────────────────────────────────────────────
st.divider()
with st.expander("Full squad table"):
    name_col = "web_name" if "web_name" in squad.columns else "name"
    show = squad[[name_col, "team", "position", "price", xpts_col]].copy()
    show = show.rename(columns={name_col: "name", xpts_col: "xPts"})
    show["role"] = ["Starter"] * len(solution.starters) + ["Bench"] * len(solution.bench)
    st.dataframe(
        show.style.format({"price": "£{:.1f}m", "xPts": "{:.1f}"}),
        use_container_width=True,
    )
