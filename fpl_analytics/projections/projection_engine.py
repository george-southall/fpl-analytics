"""Orchestrate full player projections across upcoming gameweeks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_analytics.config import settings
from fpl_analytics.models.dixon_coles import DixonColesModel
from fpl_analytics.models.score_matrix import ScoreMatrix
from fpl_analytics.projections.fixture_difficulty import compute_fixture_difficulty
from fpl_analytics.projections.minutes_model import compute_xmins
from fpl_analytics.projections.points_calculator import calculate_player_xpts
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


def _build_per90_rates(
    players_df: pd.DataFrame,
    understat_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge Understat per-90 rates onto FPL players by name.

    Falls back to FPL-derived rates when Understat data is unavailable.
    """
    if understat_df is not None and len(understat_df) > 0:
        # Use the most recent season's data per player
        latest = (
            understat_df.sort_values("season", ascending=False)
            .drop_duplicates(subset=["player_name"], keep="first")
            [["player_name", "xg_per_90", "xa_per_90", "matches", "minutes"]]
        )
        # Attempt to join on web_name (best effort; won't catch all)
        merged = players_df.merge(
            latest,
            left_on="web_name",
            right_on="player_name",
            how="left",
        )
    else:
        merged = players_df.copy()
        merged["xg_per_90"] = None
        merged["xa_per_90"] = None
        merged["matches"] = None

    # Fall back to FPL-derived xG/xA for players not in Understat
    total_mins = merged["minutes"].fillna(1).clip(lower=1)
    merged["xg_per_90"] = merged["xg_per_90"].fillna(
        merged.get("expected_goals", 0) / total_mins * 90
    )
    merged["xa_per_90"] = merged["xa_per_90"].fillna(
        merged.get("expected_assists", 0) / total_mins * 90
    )

    # Bonus per 90 from FPL history
    merged["bonus_per_90"] = merged.get("bonus", 0) / total_mins * 90

    # GK saves per 90
    merged["saves_per_90"] = merged.get("saves", 0) / total_mins * 90

    return merged


def run_projections(
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    model: DixonColesModel,
    history_map: dict[int, pd.DataFrame] | None = None,
    understat_df: pd.DataFrame | None = None,
    upcoming_gws: list[int] | None = None,
    dgw_ids: set[int] | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Produce a full player projection table.

    Args:
        players_df: Master FPL player DataFrame.
        fixtures_df: FPL fixture DataFrame with team names.
        model: Fitted DixonColesModel.
        history_map: player_id → GW history DataFrame (for xMins calculation).
        understat_df: Optional Understat xG/xA data.
        upcoming_gws: GW numbers to project. If None, derived from fixtures.
        dgw_ids: Player IDs with a double gameweek.
        horizon: Number of GWs to project (default: settings.projection_horizon_gws).

    Returns:
        DataFrame with one row per player; columns per GW + totals.
    """
    horizon = horizon or settings.projection_horizon_gws
    history_map = history_map or {}
    dgw_ids = dgw_ids or set()

    # ── Determine upcoming GWs ────────────────────────────────────────────────
    if upcoming_gws is None:
        pending = fixtures_df[~fixtures_df["finished"].astype(bool)]
        if "event" in pending.columns:
            all_gws = sorted(pending["event"].dropna().unique().astype(int))
            upcoming_gws = all_gws[:horizon]
        else:
            upcoming_gws = []

    logger.info(f"Projecting {len(players_df)} players over GWs: {upcoming_gws}")

    # ── Merge in per-90 rates ─────────────────────────────────────────────────
    players = _build_per90_rates(players_df, understat_df)

    # ── Compute expected minutes for each player (baseline GW) ───────────────
    players = compute_xmins(players, history_map, dgw_ids=dgw_ids)

    # ── Fixture difficulty for all upcoming GWs ───────────────────────────────
    difficulty_df = compute_fixture_difficulty(fixtures_df, model, upcoming_gws)

    # ── Build per-GW predictions ──────────────────────────────────────────────
    predictor = ScoreMatrix(model)
    known_teams = set(model.params.teams)

    gw_cols: list[str] = []
    gw_detail: dict[str, list] = {}  # gw_label → list of xpts per player

    for gw in upcoming_gws:
        gw_label = f"GW{gw}_xPts"
        gw_cols.append(gw_label)
        gw_detail[gw_label] = []

        # Get fixtures for this GW
        gw_fixes = fixtures_df[
            (fixtures_df["event"] == gw) & (~fixtures_df["finished"].astype(bool))
        ]

        for _, player in players.iterrows():
            team = player.get("team_name", "")
            xmins = float(player.get("xMins", 0))

            if xmins <= 0 or team not in known_teams:
                gw_detail[gw_label].append(0.0)
                continue

            # Find this player's fixture in this GW
            fix = gw_fixes[
                (gw_fixes["team_h_name"] == team) | (gw_fixes["team_a_name"] == team)
            ]

            if fix.empty:
                # Blank gameweek for this team
                gw_detail[gw_label].append(0.0)
                continue

            fix_row = fix.iloc[0]
            is_home = fix_row["team_h_name"] == team
            opponent = fix_row["team_a_name"] if is_home else fix_row["team_h_name"]

            if opponent not in known_teams:
                gw_detail[gw_label].append(0.0)
                continue

            # Get match prediction
            if is_home:
                pred = predictor.predict(team, opponent)
                team_xg = pred.home_xg
                team_xgc = pred.away_xg
                cs_prob = pred.home_cs_prob
            else:
                pred = predictor.predict(opponent, team)
                team_xg = pred.away_xg
                team_xgc = pred.home_xg
                cs_prob = pred.away_cs_prob

            # Apply confidence decay for further-out GWs
            gw_idx = upcoming_gws.index(gw)
            decay = 0.95 ** gw_idx

            breakdown = calculate_player_xpts(
                player_id=int(player["id"]),
                name=player["web_name"],
                position=player["position"],
                xmins=xmins * decay,
                xg_per90=float(player.get("xg_per_90", 0) or 0),
                xa_per90=float(player.get("xa_per_90", 0) or 0),
                team_xg=team_xg,
                team_xgc=team_xgc,
                cs_prob=cs_prob,
                saves_per90=float(player.get("saves_per_90", 0) or 0),
                bonus_per90=float(player.get("bonus_per_90", 0) or 0),
            )
            gw_detail[gw_label].append(round(breakdown.xPts_total, 2))

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    result = players[
        ["id", "web_name", "team_name", "position", "price",
         "selected_by_percent", "xMins", "xg_per_90", "xa_per_90"]
    ].copy()
    result = result.rename(columns={"web_name": "name", "team_name": "team"})

    for gw_label in gw_cols:
        result[gw_label] = gw_detail[gw_label]

    if gw_cols:
        result["total_xPts"] = result[gw_cols].sum(axis=1)
    else:
        result["total_xPts"] = 0.0

    result = result.sort_values("total_xPts", ascending=False).reset_index(drop=True)

    logger.info(
        f"Projections complete. Top player: {result.iloc[0]['name']} "
        f"({result.iloc[0]['total_xPts']:.1f} pts)"
    )
    return result


def save_projections(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save projections to CSV."""
    path = path or settings.data_dir / "processed" / "projections.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved projections to {path}")
    return path
