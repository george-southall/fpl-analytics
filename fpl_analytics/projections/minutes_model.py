"""Expected minutes model for FPL players."""

from __future__ import annotations

import pandas as pd

from fpl_analytics.config import settings
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

# Availability multipliers based on FPL status code
STATUS_AVAILABILITY = {
    "a": 1.00,  # available
    "d": 0.50,  # doubtful
    "u": 0.00,  # unavailable
    "s": 0.00,  # suspended
    "i": 0.00,  # injured
    "n": 0.75,  # not in squad (may rotate in)
}

# Chance-of-playing override: if FPL gives a percentage, use it directly
# (FPL API provides null, 75, 50, 25, or 0)


def _availability_factor(row: pd.Series) -> float:
    """Combine status + chance_of_playing into a single availability scalar."""
    cop = row.get("chance_of_playing_next_round")
    if cop is not None and not pd.isna(cop):
        return float(cop) / 100.0
    return STATUS_AVAILABILITY.get(row.get("status", "a"), 1.0)


def compute_xmins(
    players_df: pd.DataFrame,
    history_map: dict[int, pd.DataFrame] | None = None,
    dgw_ids: set[int] | None = None,
    window: int | None = None,
    current_gw: int | None = None,
) -> pd.DataFrame:
    """Compute expected minutes per player for one upcoming gameweek.

    Args:
        players_df: Master player DataFrame from FPL API.
        history_map: Dict of player_id → GW history DataFrame
                     (columns: minutes, round). Fetched from element-summary.
                     If None or empty, falls back to season-average minutes.
        dgw_ids: Set of player IDs who have a double gameweek (play twice).
        window: Number of trailing GWs to average. Defaults to settings value.
        current_gw: Current gameweek number (used to compute season averages).

    Returns:
        players_df with added columns: trailing_mins, availability, xMins
    """
    window = window or settings.trailing_minutes_window
    dgw_ids = dgw_ids or set()
    history_map = history_map or {}
    gws_played = max((current_gw or 20) - 1, 1)

    rows = []
    for _, player in players_df.iterrows():
        pid = int(player["id"])
        hist = history_map.get(pid)

        if hist is not None and len(hist) > 0:
            recent = hist.sort_values("round", ascending=False).head(window)
            trailing_mins = float(recent["minutes"].mean())
        else:
            # Fall back to season-average using total minutes from the API.
            # This is far more accurate than a price-based heuristic: a player
            # with 14 season minutes correctly gets ~0.5 xMins, not 45.
            season_mins = float(player.get("minutes", 0) or 0)
            trailing_mins = season_mins / gws_played

        availability = _availability_factor(player)
        base_xmins = trailing_mins * availability

        # Double gameweek doubles expected minutes (capped at 2×90)
        if pid in dgw_ids:
            xmins = min(base_xmins * 2, 180.0)
            is_dgw = True
        else:
            xmins = base_xmins
            is_dgw = False

        rows.append(
            {
                "id": pid,
                "trailing_mins": round(trailing_mins, 1),
                "availability": round(availability, 2),
                "is_dgw": is_dgw,
                "xMins": round(xmins, 1),
            }
        )

    mins_df = pd.DataFrame(rows)
    return players_df.merge(mins_df, on="id", how="left")


def appearance_points(xmins: float) -> float:
    """Expected appearance points from expected minutes.

    For double gameweeks (xmins > 90), the player can earn appearance points
    in both matches, so we split into two separate game calculations.
    """
    if xmins <= 0:
        return 0.0
    if xmins > 90:
        # DGW: first game (guaranteed 90 mins if played) + remaining mins
        return appearance_points(90.0) + appearance_points(xmins - 90.0)
    p_any = min(xmins / 90.0, 1.0)
    p_ge60 = max(min((xmins - 30) / 60.0, 1.0), 0.0)
    p_lt60 = p_any - p_ge60
    return p_ge60 * 2.0 + p_lt60 * 1.0
