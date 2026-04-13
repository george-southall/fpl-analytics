"""FPL Analytics — Streamlit dashboard entry point.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="FPL Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ FPL Analytics Engine")
st.markdown(
    """
    A professional-grade Fantasy Premier League projection and optimisation platform
    built on the **Dixon-Coles Poisson regression model**.

    Use the sidebar to navigate between pages.
    """
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/01_player_projections.py", label="📊 Player Projections", use_container_width=True)
with col2:
    st.page_link("pages/02_fixture_difficulty.py", label="📅 Fixture Difficulty", use_container_width=True)
with col3:
    st.page_link("pages/03_team_strengths.py", label="💪 Team Strengths", use_container_width=True)
with col4:
    st.page_link("pages/04_squad_optimiser.py", label="🧮 Squad Optimiser", use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.page_link("pages/05_transfer_planner.py", label="🔄 Transfer Planner", use_container_width=True)
with col6:
    st.page_link("pages/06_price_changes.py", label="💰 Price Change Alerts", use_container_width=True)

st.divider()
st.caption(
    "Data sources: FPL Official API · football-data.co.uk · Understat  |  "
    "Model: Dixon-Coles (1997) Poisson regression"
)
