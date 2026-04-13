"""Captain and vice-captain selection by expected value."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CaptainRecommendation:
    """Captain and vice-captain recommendation with EV breakdown."""

    captain: str
    captain_id: int
    captain_xpts: float
    captain_ev_bonus: float  # net captain bonus = xPts (doubles score)
    captain_ownership: float

    vice_captain: str
    vice_captain_id: int
    vice_captain_xpts: float
    vice_captain_ownership: float

    differential_captain: str | None
    differential_captain_id: int | None
    differential_captain_xpts: float | None
    differential_captain_ownership: float | None

    all_candidates: pd.DataFrame


def pick_captain(
    squad: pd.DataFrame,
    xpts_col: str = "total_xPts",
    ownership_col: str = "selected_by_percent",
    differential_threshold: float = 10.0,
) -> CaptainRecommendation:
    """Select captain and vice-captain by highest expected value.

    Captain EV bonus = xPts (the extra points from doubling).
    So the best captain is simply the player with highest projected points.

    Args:
        squad: Squad DataFrame with xpts and ownership columns.
        xpts_col: Projected points column.
        ownership_col: Ownership percentage column.
        differential_threshold: Max ownership % to qualify as differential.

    Returns:
        CaptainRecommendation with captain, vice, and differential picks.
    """
    df = squad.copy().sort_values(xpts_col, ascending=False).reset_index(drop=True)

    # Captain = highest xPts
    cap = df.iloc[0]

    # Vice = second highest xPts
    vice = df.iloc[1] if len(df) > 1 else cap

    # Differential = highest xPts among low-ownership players
    low_ownership = df[df[ownership_col] < differential_threshold]
    if not low_ownership.empty:
        diff = low_ownership.iloc[0]
        diff_name = diff.get("web_name") or diff.get("name", "")
        diff_id = int(diff["id"])
        diff_xpts = float(diff[xpts_col])
        diff_ownership = float(diff[ownership_col])
    else:
        diff_name = None
        diff_id = None
        diff_xpts = None
        diff_ownership = None

    # Build candidate table for display
    candidates = df[["id", ownership_col, xpts_col]].copy()
    name_col = "web_name" if "web_name" in df.columns else "name"
    candidates.insert(0, "name", df[name_col])
    candidates["captain_ev"] = candidates[xpts_col]  # net bonus = xPts itself
    candidates = candidates.head(10)

    cap_name = cap.get("web_name") or cap.get("name", "")
    vice_name = vice.get("web_name") or vice.get("name", "")

    logger.info(f"Captain: {cap_name} ({float(cap[xpts_col]):.1f} xPts)")
    logger.info(f"Vice-captain: {vice_name} ({float(vice[xpts_col]):.1f} xPts)")
    if diff_name:
        logger.info(
            f"Differential: {diff_name} ({diff_xpts:.1f} xPts, "
            f"{diff_ownership:.1f}% owned)"
        )

    return CaptainRecommendation(
        captain=cap_name,
        captain_id=int(cap["id"]),
        captain_xpts=round(float(cap[xpts_col]), 2),
        captain_ev_bonus=round(float(cap[xpts_col]), 2),
        captain_ownership=round(float(cap[ownership_col]), 1),
        vice_captain=vice_name,
        vice_captain_id=int(vice["id"]),
        vice_captain_xpts=round(float(vice[xpts_col]), 2),
        vice_captain_ownership=round(float(vice[ownership_col]), 1),
        differential_captain=diff_name,
        differential_captain_id=diff_id,
        differential_captain_xpts=round(diff_xpts, 2) if diff_xpts else None,
        differential_captain_ownership=round(diff_ownership, 1) if diff_ownership else None,
        all_candidates=candidates,
    )


def format_captain_recommendation(rec: CaptainRecommendation) -> str:
    """Format captain recommendation for CLI display."""
    lines = [
        "\n" + "=" * 50,
        "  CAPTAIN RECOMMENDATION",
        "=" * 50,
        f"  ★ Captain:      {rec.captain:<20} {rec.captain_xpts:.1f} xPts  "
        f"({rec.captain_ownership:.1f}% owned)",
        f"  ☆ Vice-captain: {rec.vice_captain:<20} {rec.vice_captain_xpts:.1f} xPts  "
        f"({rec.vice_captain_ownership:.1f}% owned)",
    ]
    if rec.differential_captain:
        lines.append(
            f"  ◆ Differential: {rec.differential_captain:<20} "
            f"{rec.differential_captain_xpts:.1f} xPts  "
            f"({rec.differential_captain_ownership:.1f}% owned)"
        )
    lines.append("=" * 50)
    return "\n".join(lines)
