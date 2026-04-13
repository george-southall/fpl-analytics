"""Fetch historical Premier League results from football-data.co.uk."""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from fpl_analytics.config import settings
from fpl_analytics.db import HistoricalResult, get_session, init_db
from fpl_analytics.utils.fpl_constants import SEASON_CODES, normalise_team_name
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


def _build_url(season_code: str) -> str:
    """Build download URL for a given season code like '2425'."""
    return f"{settings.football_data_base_url}/{season_code}/E0.csv"


def fetch_season(season_code: str) -> pd.DataFrame:
    """Download and parse one season of PL results."""
    url = _build_url(season_code)
    logger.info(f"Fetching results for season {season_code}: {url}")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip")

    required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Season {season_code} CSV missing columns: {missing}")

    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    df = df.rename(
        columns={
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "Date": "date",
        }
    )

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["season"] = season_code

    # Normalise team names
    df["home_team"] = df["home_team"].apply(normalise_team_name)
    df["away_team"] = df["away_team"].apply(normalise_team_name)

    # Parse dates — football-data.co.uk uses DD/MM/YYYY or DD/MM/YY
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(f"  Season {season_code}: {len(df)} matches")
    return df


def fetch_all_seasons(n_seasons: int | None = None) -> pd.DataFrame:
    """Download and concatenate multiple seasons of results."""
    n = n_seasons or settings.seasons_to_fetch
    codes = SEASON_CODES[:n]
    frames = []

    for code in codes:
        try:
            df = fetch_season(code)
            frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch season {code}: {e}")

    if not frames:
        raise RuntimeError("Could not fetch any season data")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    logger.info(f"Total historical results: {len(combined)} matches across {len(frames)} seasons")
    return combined


def apply_time_decay(df: pd.DataFrame, xi: float | None = None) -> pd.DataFrame:
    """Apply exponential time decay weighting to results.

    Weight = exp(-xi * days_since_match) where xi controls how fast old matches fade.
    Default xi ≈ 0.0065 means a match 6 months ago has ~half the weight.
    """
    xi = xi or settings.dc_time_decay_xi
    dates = pd.to_datetime(df["date"])
    today = pd.Timestamp(datetime.today().date())
    days_since = (today - dates).dt.days.astype(float)
    df = df.copy()
    df["weight"] = np.exp(-xi * days_since)
    return df


def persist_results(df: pd.DataFrame) -> int:
    """Persist historical results to the database."""
    init_db()
    session = get_session()
    try:
        session.query(HistoricalResult).delete()
        for _, row in df.iterrows():
            session.add(
                HistoricalResult(
                    season=row["season"],
                    date=row["date"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    home_goals=row["home_goals"],
                    away_goals=row["away_goals"],
                )
            )
        session.commit()
        logger.info(f"Persisted {len(df)} historical results to database")
        return len(df)
    finally:
        session.close()
