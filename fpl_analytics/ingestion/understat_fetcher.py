"""Fetch xG/xA data from Understat."""

from __future__ import annotations

import asyncio

import pandas as pd
from understat import Understat

from fpl_analytics.db import UnderstatPlayer, get_session, init_db
from fpl_analytics.utils.fpl_constants import normalise_team_name
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

# Understat uses full season year (e.g. 2024 for the 2024/25 season)
UNDERSTAT_SEASONS = [2024, 2023, 2022, 2021, 2020]


async def _fetch_league_players(season: int) -> list[dict]:
    """Fetch all PL player stats for a given season from Understat."""
    async with Understat() as understat:
        players = await understat.get_league_players("epl", season)
    return players


def fetch_season(season: int) -> pd.DataFrame:
    """Fetch player-level xG/xA for one season."""
    logger.info(f"Fetching Understat data for season {season}")

    players = asyncio.run(_fetch_league_players(season))

    rows = []
    for p in players:
        games = int(p.get("games", 0))
        minutes = float(p.get("time", 0))
        per_90 = minutes / 90.0 if minutes > 0 else 0.0

        xg = float(p.get("xG", 0))
        xa = float(p.get("xA", 0))

        rows.append(
            {
                "player_name": p.get("player_name", ""),
                "team": normalise_team_name(p.get("team_title", "")),
                "season": str(season),
                "matches": games,
                "minutes": minutes,
                "goals": int(p.get("goals", 0)),
                "xg": xg,
                "assists": int(p.get("assists", 0)),
                "xa": xa,
                "npg": int(p.get("npg", 0)),
                "npxg": float(p.get("npxG", 0)),
                "xg_per_90": xg / per_90 if per_90 > 0 else 0.0,
                "xa_per_90": xa / per_90 if per_90 > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(f"  Season {season}: {len(df)} players")
    return df


def fetch_all_seasons(seasons: list[int] | None = None) -> pd.DataFrame:
    """Fetch xG/xA data for multiple seasons."""
    seasons = seasons or UNDERSTAT_SEASONS
    frames = []

    for season in seasons:
        try:
            df = fetch_season(season)
            frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch Understat season {season}: {e}")

    if not frames:
        raise RuntimeError("Could not fetch any Understat data")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total Understat records: {len(combined)} across {len(frames)} seasons")
    return combined


def fetch_current_season() -> pd.DataFrame:
    """Fetch only the current season's data (most common use case)."""
    return fetch_season(UNDERSTAT_SEASONS[0])


def persist_understat(df: pd.DataFrame) -> int:
    """Persist Understat player data to the database."""
    init_db()
    session = get_session()
    try:
        session.query(UnderstatPlayer).delete()
        for _, row in df.iterrows():
            session.add(
                UnderstatPlayer(
                    player_name=row["player_name"],
                    team=row["team"],
                    season=row["season"],
                    matches=row["matches"],
                    goals=row["goals"],
                    xg=row["xg"],
                    assists=row["assists"],
                    xa=row["xa"],
                    npg=row["npg"],
                    npxg=row["npxg"],
                    xg_per_90=row["xg_per_90"],
                    xa_per_90=row["xa_per_90"],
                )
            )
        session.commit()
        logger.info(f"Persisted {len(df)} Understat player records to database")
        return len(df)
    finally:
        session.close()
