"""Fixture difficulty ratings derived from Dixon-Coles team strengths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_analytics.models.dixon_coles import DixonColesModel
from fpl_analytics.models.score_matrix import ScoreMatrix
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


def _xg_to_difficulty(xg_for: float, thresholds: list[float]) -> int:
    """Convert expected goals into a 1-5 difficulty rating.

    Higher xG_for = easier fixture (1).  Lower xG_for = harder fixture (5).
    Thresholds are the 20th, 40th, 60th, 80th percentiles of xG_for across
    all fixtures in the upcoming window, so the scale auto-calibrates each season.
    """
    for i, thresh in enumerate(thresholds):
        if xg_for >= thresh:
            return i + 1
    return 5


def compute_fixture_difficulty(
    fixtures_df: pd.DataFrame,
    model: DixonColesModel,
    upcoming_gws: list[int] | None = None,
) -> pd.DataFrame:
    """Compute fixture difficulty metrics for all upcoming fixtures.

    For each fixture, computes from both teams' perspectives:
    - Expected goals for and against
    - Clean sheet probability
    - Difficulty rating (1-5, auto-calibrated from xG distribution)
    - is_dgw flag (True when the team plays twice in that GW)

    Unknown opponents (newly promoted teams not yet in the model) are recorded
    with difficulty=3 (neutral) so the team is not incorrectly shown as blank.

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

    # ── First pass: compute xG for all known-team fixtures ────────────────────
    # We need the full distribution to set percentile thresholds.
    xg_values: list[float] = []
    fixture_data: list[dict] = []

    for _, fix in pending.iterrows():
        h = fix["team_h_name"]
        a = fix["team_a_name"]
        gw = fix["event"]

        h_known = h in known_teams
        a_known = a in known_teams

        if h_known and a_known:
            pred = predictor.predict(h, a)
            xg_values.extend([pred.home_xg, pred.away_xg])
            fixture_data.append({
                "h": h, "a": a, "gw": gw,
                "home_xg": pred.home_xg, "away_xg": pred.away_xg,
                "home_cs": pred.home_cs_prob, "away_cs": pred.away_cs_prob,
                "home_win": pred.home_win_prob, "draw": pred.draw_prob,
                "away_win": pred.away_win_prob,
                "h_known": True, "a_known": True,
            })
        else:
            logger.debug(f"  {h} vs {a} GW{gw}: one team not in model — using defaults")
            fixture_data.append({
                "h": h, "a": a, "gw": gw,
                "home_xg": None, "away_xg": None,
                "home_cs": None, "away_cs": None,
                "home_win": None, "draw": None, "away_win": None,
                "h_known": h_known, "a_known": a_known,
            })

    # ── Compute percentile thresholds from known-fixture xG distribution ──────
    if xg_values:
        thresholds = [
            float(np.percentile(xg_values, 80)),  # top 20% → difficulty 1
            float(np.percentile(xg_values, 60)),
            float(np.percentile(xg_values, 40)),
            float(np.percentile(xg_values, 20)),  # bottom 20% → difficulty 5
        ]
    else:
        thresholds = [1.6, 1.3, 1.1, 0.9]  # sensible PL defaults

    # ── Second pass: build rows with calibrated difficulty ────────────────────
    rows = []
    for fd in fixture_data:
        h, a, gw = fd["h"], fd["a"], fd["gw"]

        if fd["home_xg"] is not None:
            rows.append({
                "team": h, "opponent": a, "venue": "H", "gw": int(gw),
                "xg_for": round(fd["home_xg"], 3),
                "xg_against": round(fd["away_xg"], 3),
                "cs_prob": round(fd["home_cs"], 3),
                "difficulty": _xg_to_difficulty(fd["home_xg"], thresholds),
                "win_prob": round(fd["home_win"], 3),
                "draw_prob": round(fd["draw"], 3),
                "loss_prob": round(fd["away_win"], 3),
            })
            rows.append({
                "team": a, "opponent": h, "venue": "A", "gw": int(gw),
                "xg_for": round(fd["away_xg"], 3),
                "xg_against": round(fd["home_xg"], 3),
                "cs_prob": round(fd["away_cs"], 3),
                "difficulty": _xg_to_difficulty(fd["away_xg"], thresholds),
                "win_prob": round(fd["away_win"], 3),
                "draw_prob": round(fd["draw"], 3),
                "loss_prob": round(fd["home_win"], 3),
            })
        else:
            # One team unknown — record fixture with neutral difficulty=3
            if fd["h_known"]:
                rows.append({
                    "team": h, "opponent": a, "venue": "H", "gw": int(gw),
                    "xg_for": None, "xg_against": None, "cs_prob": None,
                    "difficulty": 3, "win_prob": None, "draw_prob": None, "loss_prob": None,
                })
            if fd["a_known"]:
                rows.append({
                    "team": a, "opponent": h, "venue": "A", "gw": int(gw),
                    "xg_for": None, "xg_against": None, "cs_prob": None,
                    "difficulty": 3, "win_prob": None, "draw_prob": None, "loss_prob": None,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Flag double gameweeks ─────────────────────────────────────────────────
    fixture_counts = df.groupby(["team", "gw"])["venue"].count().reset_index()
    fixture_counts = fixture_counts.rename(columns={"venue": "n_fixtures"})
    df = df.merge(fixture_counts, on=["team", "gw"], how="left")
    df["is_dgw"] = df["n_fixtures"] > 1

    df = df.sort_values(["team", "gw"]).reset_index(drop=True)
    return df


def fixture_difficulty_calendar(
    difficulty_df: pd.DataFrame,
    gws: list[int],
) -> pd.DataFrame:
    """Pivot fixture difficulty into a calendar: teams × GWs.

    Double gameweeks are indicated with the minimum difficulty of the two fixtures.

    Returns:
        DataFrame with teams as rows and GWs as columns; cell = difficulty (1-5).
    """
    pivot = difficulty_df.pivot_table(
        index="team",
        columns="gw",
        values="difficulty",
        aggfunc="min",  # DGW: use the easier fixture for summary
    )
    existing = [gw for gw in gws if gw in pivot.columns]
    pivot = pivot[existing]
    pivot.columns = [f"GW{gw}" for gw in pivot.columns]
    return pivot.reset_index()
