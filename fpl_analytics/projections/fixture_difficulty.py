"""Fixture difficulty ratings derived from Dixon-Coles team strengths."""

from __future__ import annotations

import pandas as pd

from fpl_analytics.models.dixon_coles import DixonColesModel
from fpl_analytics.models.score_matrix import ScoreMatrix
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

# Difficulty scale thresholds based on opponent defence rating
# Higher opponent defence = harder for attacking; lower = easier
_DIFFICULTY_THRESHOLDS = [0.80, 0.95, 1.05, 1.20]  # defence rating breakpoints


def _defence_to_difficulty(opponent_defence: float) -> int:
    """Convert opponent defence rating to a 1-5 difficulty scale.

    Lower opponent defence = weaker defence = easier fixture (1).
    Higher opponent defence = stronger defence = harder fixture (5).
    """
    for i, thresh in enumerate(_DIFFICULTY_THRESHOLDS, start=1):
        if opponent_defence < thresh:
            return i
    return 5


def compute_fixture_difficulty(
    fixtures_df: pd.DataFrame,
    model: DixonColesModel,
    upcoming_gws: list[int] | None = None,
) -> pd.DataFrame:
    """Compute fixture difficulty metrics for all upcoming fixtures.

    For each fixture, computes from both the home and away team's perspective:
    - Expected goals for and against
    - Clean sheet probability
    - Difficulty rating (1-5)

    Args:
        fixtures_df: DataFrame with columns: event, team_h_name, team_a_name, finished.
        model: Fitted DixonColesModel.
        upcoming_gws: List of GW numbers to include. If None, uses all unfinished.

    Returns:
        DataFrame with one row per team per fixture, sorted by team and GW.
    """
    predictor = ScoreMatrix(model)
    known_teams = set(model.params.teams)

    pending = fixtures_df[~fixtures_df["finished"].astype(bool)].copy()
    if upcoming_gws:
        pending = pending[pending["event"].isin(upcoming_gws)]

    rows = []
    for _, fix in pending.iterrows():
        h = fix["team_h_name"]
        a = fix["team_a_name"]
        gw = fix["event"]

        if h not in known_teams or a not in known_teams:
            logger.debug(f"  Skipping {h} vs {a} — team(s) not in model")
            continue

        pred = predictor.predict(h, a)

        # Home team perspective
        rows.append({
            "team": h,
            "opponent": a,
            "venue": "H",
            "gw": int(gw) if pd.notna(gw) else None,
            "xg_for": pred.home_xg,
            "xg_against": pred.away_xg,
            "cs_prob": pred.home_cs_prob,
            "difficulty": _defence_to_difficulty(
                model.params.defence[model._team_idx[a]]
            ),
            "win_prob": pred.home_win_prob,
            "draw_prob": pred.draw_prob,
            "loss_prob": pred.away_win_prob,
        })

        # Away team perspective
        rows.append({
            "team": a,
            "opponent": h,
            "venue": "A",
            "gw": int(gw) if pd.notna(gw) else None,
            "xg_for": pred.away_xg,
            "xg_against": pred.home_xg,
            "cs_prob": pred.away_cs_prob,
            "difficulty": _defence_to_difficulty(
                model.params.defence[model._team_idx[h]]
            ),
            "win_prob": pred.away_win_prob,
            "draw_prob": pred.draw_prob,
            "loss_prob": pred.home_win_prob,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["team", "gw"]).reset_index(drop=True)
    return df


def fixture_difficulty_calendar(
    difficulty_df: pd.DataFrame,
    gws: list[int],
) -> pd.DataFrame:
    """Pivot fixture difficulty into a calendar: teams × GWs.

    Args:
        difficulty_df: Output of compute_fixture_difficulty.
        gws: Ordered list of GW numbers to show as columns.

    Returns:
        DataFrame with teams as rows and GWs as columns; cell = difficulty (1-5).
    """
    pivot = difficulty_df.pivot_table(
        index="team",
        columns="gw",
        values="difficulty",
        aggfunc="min",  # if DGW, take the easier fixture
    )
    # Keep only requested GWs and fill missing with NaN
    existing = [gw for gw in gws if gw in pivot.columns]
    pivot = pivot[existing]
    pivot.columns = [f"GW{gw}" for gw in pivot.columns]
    return pivot.reset_index()
