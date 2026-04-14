"""Cached data loading for the Streamlit dashboard.

All expensive operations (model fitting, API calls, projections) are wrapped
in st.cache_resource or st.cache_data so they only run once per session.
"""

from __future__ import annotations

import streamlit as st


# ── Model & data (heavy, cached for session lifetime) ─────────────────────────

@st.cache_resource(show_spinner="Fitting Dixon-Coles model…")
def load_model():
    """Fit Dixon-Coles on historical results, then register any current PL teams
    that have no historical data (e.g. newly promoted sides like Sunderland)."""
    from fpl_analytics.ingestion.fpl_api import FPLClient
    from fpl_analytics.ingestion.results_fetcher import apply_time_decay, fetch_all_seasons
    from fpl_analytics.models.team_strengths import TeamStrengths

    results_df = fetch_all_seasons()
    results_df = apply_time_decay(results_df)
    ts = TeamStrengths()
    ts.fit(results_df)

    # Inject any current PL team missing from the model with average parameters
    try:
        current_teams = FPLClient().get_teams_df()["name"].tolist()
        for team in current_teams:
            ts.model.register_team(team)
    except Exception:
        pass  # non-critical: fixtures for unknown teams already default to difficulty 3

    return ts


@st.cache_data(ttl=3600, show_spinner="Loading FPL data…")
def load_fpl_data():
    """Load players, teams, and fixtures from FPL API (refreshed hourly)."""
    from fpl_analytics.db import init_db
    from fpl_analytics.ingestion.fpl_api import FPLClient

    init_db()
    client = FPLClient()
    players = client.get_players_df()
    teams = client.get_teams_df()
    fixtures = client.get_fixtures_df()
    gameweeks = client.get_gameweeks_df()
    return players, teams, fixtures, gameweeks


@st.cache_data(ttl=3600, show_spinner="Computing projections…")
def load_projections(horizon: int = 6):
    """Run the full projection engine. Cached per horizon value."""
    from fpl_analytics.projections.projection_engine import run_projections

    players, _, fixtures, _ = load_fpl_data()
    ts = load_model()
    return run_projections(players, fixtures, ts.model, horizon=horizon)


@st.cache_data(ttl=3600, show_spinner="Computing fixture difficulty…")
def load_fixture_difficulty(upcoming_gws: tuple[int, ...]):
    """Compute fixture difficulty for the given GWs."""
    from fpl_analytics.projections.fixture_difficulty import compute_fixture_difficulty

    _, _, fixtures, _ = load_fpl_data()
    ts = load_model()
    return compute_fixture_difficulty(fixtures, ts.model, list(upcoming_gws))


# ── Helpers ───────────────────────────────────────────────────────────────────

def upcoming_gw_list(n: int = 6) -> list[int]:
    """Return the next n unfinished GW numbers."""
    _, _, fixtures, _ = load_fpl_data()
    pending = fixtures[~fixtures["finished"].astype(bool)]
    gws = sorted(pending["event"].dropna().unique().astype(int))
    return list(gws[:n])


def get_current_gw() -> int:
    """Return the current/next gameweek number."""
    gws = upcoming_gw_list(1)
    return gws[0] if gws else 38


@st.cache_resource(show_spinner="Training price change model…")
def load_price_model(force_retrain: bool = False):
    """Load or train the XGBoost price change model. Cached for the session."""
    from fpl_analytics.price_changes.alerts import load_or_train_model

    players, _, _, _ = load_fpl_data()
    return load_or_train_model(players, force_retrain=force_retrain)


@st.cache_data(ttl=1800, show_spinner="Generating price alerts…")
def load_price_alerts():
    """Return price alert DataFrame (refreshed every 30 min)."""
    from fpl_analytics.price_changes.alerts import generate_alerts

    players, _, _, _ = load_fpl_data()
    model = load_price_model()
    return generate_alerts(players, model=model)


POSITION_COLOURS = {"GK": "#f6c90e", "DEF": "#00c2a8", "MID": "#6ecbf5", "FWD": "#e84855"}
DIFFICULTY_COLOURS = {1: "#00c2a8", 2: "#a3d977", 3: "#f6c90e", 4: "#f4845f", 5: "#e84855"}
