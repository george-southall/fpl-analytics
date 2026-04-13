"""Convert match-level probabilities into expected FPL points per player."""

from __future__ import annotations

from dataclasses import dataclass

from fpl_analytics.projections.minutes_model import appearance_points
from fpl_analytics.utils.fpl_constants import CS_POINTS, GOAL_POINTS, POINTS


@dataclass
class PlayerPointsBreakdown:
    """Expected FPL points broken down by source."""

    player_id: int
    name: str
    position: str
    xMins: float

    xPts_appearance: float
    xPts_goals: float
    xPts_assists: float
    xPts_clean_sheet: float
    xPts_goals_conceded: float
    xPts_saves: float
    xPts_bonus: float
    xPts_cards: float
    xPts_total: float

    # Constituent inputs (for inspection)
    xGoals: float
    xAssists: float
    xCS: float
    xGC: float
    xSaves: float
    xBonus: float


def calculate_player_xpts(
    player_id: int,
    name: str,
    position: str,
    xmins: float,
    xg_per90: float,
    xa_per90: float,
    team_xg: float,
    team_xgc: float,
    cs_prob: float,
    saves_per90: float = 0.0,
    bonus_per90: float = 0.0,
    yellow_card_prob: float = 0.03,
    red_card_prob: float = 0.003,
    own_goal_rate: float = 0.002,
) -> PlayerPointsBreakdown:
    """Compute expected FPL points for one player for one fixture.

    Strategy:
    - Scale player's per-90 xG/xA rates by team-level expected goals from
      Dixon-Coles, so the individual totals are consistent with match forecast.
    - Clean sheet probability comes directly from the score matrix.
    - Goals conceded penalty uses expected goals against from the score matrix.

    Args:
        player_id: FPL player ID.
        name: Player display name.
        position: 'GK', 'DEF', 'MID', or 'FWD'.
        xmins: Expected minutes in this fixture.
        xg_per90: Player's xG per 90 mins (from Understat, current season).
        xa_per90: Player's xA per 90 mins.
        team_xg: Team-level expected goals FOR this fixture (from Dixon-Coles).
        team_xgc: Team-level expected goals AGAINST this fixture.
        cs_prob: Clean sheet probability for this player's team (from score matrix).
        saves_per90: GK saves per 90 (only relevant for GK).
        bonus_per90: Historical bonus points per 90.
        yellow_card_prob: Per-match probability of a yellow card.
        red_card_prob: Per-match probability of a red card.
        own_goal_rate: Per-match own goal rate.

    Returns:
        PlayerPointsBreakdown with xPts_total and full breakdown.
    """
    if xmins <= 0:
        return PlayerPointsBreakdown(
            player_id=player_id, name=name, position=position, xMins=0.0,
            xPts_appearance=0.0, xPts_goals=0.0, xPts_assists=0.0,
            xPts_clean_sheet=0.0, xPts_goals_conceded=0.0, xPts_saves=0.0,
            xPts_bonus=0.0, xPts_cards=0.0, xPts_total=0.0,
            xGoals=0.0, xAssists=0.0, xCS=0.0, xGC=0.0, xSaves=0.0, xBonus=0.0,
        )

    mins_ratio = xmins / 90.0  # proportion of a full game

    # ── Appearance ────────────────────────────────────────────────────────────
    xpts_appearance = appearance_points(xmins)

    # ── Goals ─────────────────────────────────────────────────────────────────
    # Scale player per-90 rate by team xG so player totals sum to team total
    # For GK/DEF, xg_per90 is typically ~0 but we allow it
    raw_xgoals = xg_per90 * mins_ratio
    # Cap at team's expected goals (a single player can't exceed team output)
    xgoals = min(raw_xgoals, team_xg * mins_ratio)
    goal_pts = GOAL_POINTS.get(position, 4)
    xpts_goals = xgoals * goal_pts

    # ── Assists ───────────────────────────────────────────────────────────────
    raw_xassists = xa_per90 * mins_ratio
    xassists = min(raw_xassists, team_xg * mins_ratio)
    xpts_assists = xassists * POINTS["assist"]

    # ── Clean sheet ───────────────────────────────────────────────────────────
    # Weight by probability of playing enough minutes (>= 60 mins for GK/DEF)
    if position in ("GK", "DEF"):
        p_ge60 = max(min((xmins - 30) / 60.0, 1.0), 0.0)
        xcs = cs_prob * p_ge60
        xpts_cs = xcs * CS_POINTS[position]
    elif position == "MID":
        p_any = min(xmins / 90.0, 1.0)
        xcs = cs_prob * p_any
        xpts_cs = xcs * CS_POINTS["MID"]
    else:  # FWD
        xcs = 0.0
        xpts_cs = 0.0

    # ── Goals conceded (GK/DEF only) ─────────────────────────────────────────
    if position in ("GK", "DEF"):
        p_ge60 = max(min((xmins - 30) / 60.0, 1.0), 0.0)
        # Expected goals conceded scaled by minutes played
        xgc = team_xgc * (xmins / 90.0)
        # Points penalty: -1 per 2 goals conceded, only when on pitch >= 60
        xpts_gc = (xgc / 2.0) * POINTS["goals_conceded_2"] * p_ge60
    else:
        xgc = 0.0
        xpts_gc = 0.0

    # ── Saves (GK only) ───────────────────────────────────────────────────────
    if position == "GK":
        xsaves = saves_per90 * mins_ratio
        xpts_saves = (xsaves / 3.0) * POINTS["saves_3"]
    else:
        xsaves = 0.0
        xpts_saves = 0.0

    # ── Bonus ─────────────────────────────────────────────────────────────────
    xbonus = bonus_per90 * mins_ratio
    xpts_bonus = xbonus  # 1 bonus point = 1 FPL point

    # ── Cards ─────────────────────────────────────────────────────────────────
    p_plays = min(xmins / 90.0, 1.0)
    xpts_cards = (
        yellow_card_prob * p_plays * POINTS["yellow_card"]
        + red_card_prob * p_plays * POINTS["red_card"]
    )

    total = (
        xpts_appearance
        + xpts_goals
        + xpts_assists
        + xpts_cs
        + xpts_gc
        + xpts_saves
        + xpts_bonus
        + xpts_cards
    )

    return PlayerPointsBreakdown(
        player_id=player_id,
        name=name,
        position=position,
        xMins=round(xmins, 1),
        xPts_appearance=round(xpts_appearance, 3),
        xPts_goals=round(xpts_goals, 3),
        xPts_assists=round(xpts_assists, 3),
        xPts_clean_sheet=round(xpts_cs, 3),
        xPts_goals_conceded=round(xpts_gc, 3),
        xPts_saves=round(xpts_saves, 3),
        xPts_bonus=round(xpts_bonus, 3),
        xPts_cards=round(xpts_cards, 3),
        xPts_total=round(total, 3),
        xGoals=round(xgoals, 3),
        xAssists=round(xassists, 3),
        xCS=round(xcs, 3),
        xGC=round(xgc, 3),
        xSaves=round(xsaves, 3),
        xBonus=round(xbonus, 3),
    )
