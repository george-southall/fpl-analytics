"""Page 5 — Transfer Planner.

Fetches your current FPL squad and recommends optimal transfers.
"""

import pandas as pd
import streamlit as st

from dashboard.data_loader import POSITION_COLOURS, load_fpl_data, load_model
from fpl_analytics.config import settings
from fpl_analytics.optimiser.captain_picker import pick_captain
from fpl_analytics.optimiser.transfer_optimiser import optimise_transfers
from fpl_analytics.projections.projection_engine import run_projections

st.set_page_config(page_title="Transfer Planner · FPL Analytics", layout="wide")
st.title("🔄 Transfer Planner")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Team")
    team_id = st.number_input("FPL Team ID", value=settings.fpl_team_id, step=1)
    free_transfers = st.radio("Free transfers", [1, 2], horizontal=True)
    max_transfers = st.radio("Max transfers to consider", [1, 2, 3], index=1, horizontal=True)
    horizon = st.radio("Projection window (GWs)", [1, 3, 6], index=0, horizontal=True)
    itb = st.number_input("In the bank (£m)", value=0.0, step=0.1, min_value=0.0)
    run_btn = st.button("🔍 Find best transfers", type="primary", use_container_width=True)

xpts_col = "total_xPts"

if not run_btn:
    st.info(
        "Enter your FPL Team ID in the sidebar and click **Find best transfers**. "
        f"Your team ID is in the URL of your FPL team page."
    )
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data and running projections…"):
    try:
        players, _, fixtures, _ = load_fpl_data()
        ts = load_model()
        projections = run_projections(players, fixtures, ts.model, horizon=horizon)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# ── Fetch current squad ───────────────────────────────────────────────────────
with st.spinner(f"Fetching team {team_id}…"):
    try:
        from fpl_analytics.ingestion.fpl_api import FPLClient
        client = FPLClient()
        my_team_data = client.get_my_team(team_id)
        picks = my_team_data.get("picks", [])
        if not picks:
            st.error("No picks found for this team ID. Check the ID is correct.")
            st.stop()
        my_ids = {p["element"] for p in picks}
        my_squad = projections[projections["id"].isin(my_ids)].copy()
    except Exception as e:
        st.warning(f"Could not fetch live team ({e}). Using demo squad instead.")
        # Fall back to top 15 by xPts as demo
        from fpl_analytics.optimiser.squad_optimiser import optimise_squad
        sol = optimise_squad(projections, xpts_col=xpts_col)
        my_squad = pd.concat([sol.starters, sol.bench]).reset_index(drop=True)

if my_squad.empty:
    st.error("No players matched between your team and the projections.")
    st.stop()

# ── Your current squad ────────────────────────────────────────────────────────
st.subheader("Your current squad")
name_col = "web_name" if "web_name" in my_squad.columns else "name"

# Captain recommendation on current squad
cap_rec = pick_captain(my_squad, xpts_col=xpts_col)
st.markdown(
    f"**★ Captain:** {cap_rec.captain} ({cap_rec.captain_xpts:.1f} xPts)  &nbsp;&nbsp;"
    f"**☆ Vice:** {cap_rec.vice_captain} ({cap_rec.vice_captain_xpts:.1f} xPts)"
    + (f"  &nbsp;&nbsp;**◆ Differential:** {cap_rec.differential_captain} "
       f"({cap_rec.differential_captain_xpts:.1f} xPts, {cap_rec.differential_captain_ownership:.1f}% owned)"
       if cap_rec.differential_captain else ""),
    unsafe_allow_html=True,
)

# Squad display grouped by position
for pos in ["GK", "DEF", "MID", "FWD"]:
    pos_players = my_squad[my_squad["position"] == pos]
    if pos_players.empty:
        continue
    colour = POSITION_COLOURS[pos]
    cols = st.columns(len(pos_players))
    for col, (_, p) in zip(cols, pos_players.iterrows()):
        name = p.get("web_name", p.get("name", ""))
        col.markdown(
            f"<div style='background:{colour};padding:6px 8px;border-radius:6px;"
            f"text-align:center;margin:2px'>"
            f"<b style='font-size:13px'>{name}</b><br>"
            f"<span style='font-size:11px'>£{p['price']:.1f}m · {p[xpts_col]:.1f} xPts</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

squad_xpts = my_squad[xpts_col].sum()
st.caption(f"Squad total: {squad_xpts:.1f} xPts over {horizon} GW(s)")

st.divider()

# ── Transfer recommendations ──────────────────────────────────────────────────
st.subheader("Transfer recommendations")

with st.spinner("Finding best transfers…"):
    try:
        plans = optimise_transfers(
            my_squad,
            projections,
            xpts_col=xpts_col,
            n_transfers=max_transfers,
            free_transfers=free_transfers,
            budget_remaining=itb,
        )
    except Exception as e:
        st.error(f"Transfer optimisation failed: {e}")
        st.stop()

if not plans:
    st.info("No beneficial transfers found — your squad looks optimal!")
    st.stop()

for plan in plans:
    n = len(plan.transfers)
    hit_text = f" (+{plan.hits_taken} hit, -{plan.total_hit_cost:.0f} pts)" if plan.hits_taken else ""
    header = f"{'🟢' if plan.net_gain > 0 else '🔴'} {n} transfer{'s' if n>1 else ''}{hit_text} — net gain: **{plan.net_gain:+.1f} xPts**"

    with st.expander(header, expanded=(plan == plans[0])):
        for tr in plan.transfers:
            colour_out = POSITION_COLOURS.get(tr.position, "#ccc")
            c1, c2, c3 = st.columns([5, 1, 5])
            with c1:
                st.markdown(
                    f"<div style='background:#f8d7da;padding:8px 12px;border-radius:6px'>"
                    f"<b>OUT</b> &nbsp; {tr.player_out}<br>"
                    f"<span style='font-size:12px;color:#666'>£{tr.player_out_price:.1f}m · "
                    f"{tr.player_out_xpts:.1f} xPts · {tr.position}</span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown("<div style='text-align:center;padding-top:12px;font-size:20px'>→</div>",
                            unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f"<div style='background:#d4edda;padding:8px 12px;border-radius:6px'>"
                    f"<b>IN</b> &nbsp; {tr.player_in}<br>"
                    f"<span style='font-size:12px;color:#666'>£{tr.player_in_price:.1f}m · "
                    f"{tr.player_in_xpts:.1f} xPts · {tr.position} · "
                    f"<b>{tr.xpts_gain:+.1f} xPts</b></span></div>",
                    unsafe_allow_html=True,
                )
        if plan.hits_taken:
            st.warning(f"⚠️ This plan requires {plan.hits_taken} point hit(s): −{plan.total_hit_cost:.0f} pts")
