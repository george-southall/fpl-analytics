"""FPL API wrapper with SQLite caching."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from fpl_analytics.config import settings
from fpl_analytics.db import (
    CacheEntry,
    Fixture,
    Player,
    Team,
    get_session,
    init_db,
)
from fpl_analytics.utils.fpl_constants import POSITION_MAP, normalise_team_name
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

BASE = settings.fpl_base_url
TTL = timedelta(hours=settings.fpl_cache_ttl_hours)

# Rate limiting
_MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
_last_request_time: float = 0.0


def _rate_limit() -> None:
    """Respect rate limits by spacing out requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _get_cached(key: str) -> dict | None:
    """Retrieve cached API response if fresh enough."""
    session = get_session()
    try:
        entry = session.query(CacheEntry).filter_by(key=key).first()
        if entry and (datetime.utcnow() - entry.fetched_at) < TTL:
            logger.debug(f"Cache hit: {key}")
            return json.loads(entry.data)
        return None
    finally:
        session.close()


def _set_cached(key: str, data: dict) -> None:
    """Store an API response in cache."""
    session = get_session()
    try:
        entry = session.query(CacheEntry).filter_by(key=key).first()
        if entry:
            entry.data = json.dumps(data)
            entry.fetched_at = datetime.utcnow()
        else:
            entry = CacheEntry(key=key, data=json.dumps(data), fetched_at=datetime.utcnow())
            session.add(entry)
        session.commit()
    finally:
        session.close()


def _fetch(endpoint: str, use_cache: bool = True) -> dict:
    """Fetch from FPL API with caching and rate limiting."""
    if use_cache:
        cached = _get_cached(endpoint)
        if cached is not None:
            return cached

    url = f"{BASE}/{endpoint.lstrip('/')}"
    _rate_limit()
    logger.info(f"Fetching {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    _set_cached(endpoint, data)
    return data


class FPLClient:
    """Client for the Fantasy Premier League API."""

    def __init__(self) -> None:
        init_db()
        self._bootstrap: dict | None = None
        self._teams_map: dict[int, str] = {}

    def get_bootstrap(self, use_cache: bool = True) -> dict:
        """Fetch bootstrap-static data (players, teams, GW info)."""
        if self._bootstrap is None or not use_cache:
            self._bootstrap = _fetch("bootstrap-static/", use_cache=use_cache)
        return self._bootstrap

    def _build_teams_map(self) -> dict[int, str]:
        """Build mapping of team_id → canonical full team name from bootstrap."""
        if not self._teams_map:
            bootstrap = self.get_bootstrap()
            self._teams_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
        return self._teams_map

    # ── DataFrames ────────────────────────────────────────────────

    def get_players_df(self) -> pd.DataFrame:
        """All players as a clean DataFrame."""
        bootstrap = self.get_bootstrap()
        teams_map = self._build_teams_map()

        players = []
        for p in bootstrap["elements"]:
            players.append(
                {
                    "id": p["id"],
                    "web_name": p["web_name"],
                    "first_name": p["first_name"],
                    "second_name": p["second_name"],
                    "team_id": p["team"],
                    "team_name": normalise_team_name(teams_map.get(p["team"], "")),
                    "position_id": p["element_type"],
                    "position": POSITION_MAP.get(p["element_type"], "UNK"),
                    "now_cost": p["now_cost"],
                    "price": p["now_cost"] / 10,
                    "total_points": p["total_points"],
                    "minutes": p["minutes"],
                    "goals_scored": p["goals_scored"],
                    "assists": p["assists"],
                    "clean_sheets": p["clean_sheets"],
                    "selected_by_percent": float(p.get("selected_by_percent", 0)),
                    "transfers_in_event": p.get("transfers_in_event", 0),
                    "transfers_out_event": p.get("transfers_out_event", 0),
                    "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
                    "status": p.get("status", "a"),
                    "bonus": p.get("bonus", 0),
                    "saves": p.get("saves", 0),
                    "yellow_cards": p.get("yellow_cards", 0),
                    "red_cards": p.get("red_cards", 0),
                    "own_goals": p.get("own_goals", 0),
                    "penalties_missed": p.get("penalties_missed", 0),
                    "penalties_saved": p.get("penalties_saved", 0),
                    "goals_conceded": p.get("goals_conceded", 0),
                    "expected_goals": float(p.get("expected_goals", 0)),
                    "expected_assists": float(p.get("expected_assists", 0)),
                    "expected_goal_involvements": float(
                        p.get("expected_goal_involvements", 0)
                    ),
                    "expected_goals_conceded": float(
                        p.get("expected_goals_conceded", 0)
                    ),
                }
            )

        return pd.DataFrame(players)

    def get_teams_df(self) -> pd.DataFrame:
        """All teams as a DataFrame."""
        bootstrap = self.get_bootstrap()
        teams = []
        for t in bootstrap["teams"]:
            teams.append(
                {
                    "id": t["id"],
                    "name": normalise_team_name(t["name"]),
                    "full_name": t["name"],
                    "short_name": t["short_name"],
                    "strength": t.get("strength"),
                    "strength_overall_home": t.get("strength_overall_home"),
                    "strength_overall_away": t.get("strength_overall_away"),
                    "strength_attack_home": t.get("strength_attack_home"),
                    "strength_attack_away": t.get("strength_attack_away"),
                    "strength_defence_home": t.get("strength_defence_home"),
                    "strength_defence_away": t.get("strength_defence_away"),
                }
            )
        return pd.DataFrame(teams)

    def get_gameweeks_df(self) -> pd.DataFrame:
        """Gameweek info as a DataFrame."""
        bootstrap = self.get_bootstrap()
        return pd.DataFrame(bootstrap["events"])

    def get_fixtures_df(self) -> pd.DataFrame:
        """All fixtures as a DataFrame with team names."""
        data = _fetch("fixtures/")
        teams_map = self._build_teams_map()

        fixtures = []
        for f in data:
            fixtures.append(
                {
                    "id": f["id"],
                    "event": f.get("event"),
                    "team_h": f["team_h"],
                    "team_a": f["team_a"],
                    "team_h_name": normalise_team_name(teams_map.get(f["team_h"], "")),
                    "team_a_name": normalise_team_name(teams_map.get(f["team_a"], "")),
                    "team_h_score": f.get("team_h_score"),
                    "team_a_score": f.get("team_a_score"),
                    "finished": f.get("finished", False),
                    "kickoff_time": f.get("kickoff_time"),
                }
            )
        return pd.DataFrame(fixtures)

    def get_player_summary(self, player_id: int) -> dict:
        """Per-player GW history and upcoming fixtures."""
        return _fetch(f"element-summary/{player_id}/")

    def get_player_history_df(self, player_id: int) -> pd.DataFrame:
        """A player's GW-by-GW history this season."""
        data = self.get_player_summary(player_id)
        return pd.DataFrame(data.get("history", []))

    def get_player_fixtures_df(self, player_id: int) -> pd.DataFrame:
        """A player's upcoming fixtures."""
        data = self.get_player_summary(player_id)
        return pd.DataFrame(data.get("fixtures", []))

    def get_my_team(self, team_id: int | None = None) -> dict:
        """Fetch FPL team entry data."""
        tid = team_id or settings.fpl_team_id
        return _fetch(f"entry/{tid}/")

    def get_my_team_history(self, team_id: int | None = None) -> dict:
        """Fetch FPL team season history."""
        tid = team_id or settings.fpl_team_id
        return _fetch(f"entry/{tid}/history/")

    # ── Persist to DB ─────────────────────────────────────────────

    def persist_players(self) -> int:
        """Fetch players from API and persist to the players table. Returns count."""
        df = self.get_players_df()
        session = get_session()
        try:
            session.query(Player).delete()
            for _, row in df.iterrows():
                session.add(
                    Player(
                        id=row["id"],
                        web_name=row["web_name"],
                        first_name=row["first_name"],
                        second_name=row["second_name"],
                        team_id=row["team_id"],
                        team_name=row["team_name"],
                        position_id=row["position_id"],
                        position=row["position"],
                        now_cost=row["now_cost"],
                        total_points=row["total_points"],
                        minutes=row["minutes"],
                        goals_scored=row["goals_scored"],
                        assists=row["assists"],
                        clean_sheets=row["clean_sheets"],
                        selected_by_percent=row["selected_by_percent"],
                        transfers_in_event=row["transfers_in_event"],
                        transfers_out_event=row["transfers_out_event"],
                        chance_of_playing_next_round=row["chance_of_playing_next_round"],
                        status=row["status"],
                    )
                )
            session.commit()
            logger.info(f"Persisted {len(df)} players to database")
            return len(df)
        finally:
            session.close()

    def persist_teams(self) -> int:
        """Fetch teams from API and persist to the teams table."""
        df = self.get_teams_df()
        session = get_session()
        try:
            session.query(Team).delete()
            for _, row in df.iterrows():
                session.add(
                    Team(
                        id=row["id"],
                        name=row["name"],
                        short_name=row["short_name"],
                        strength=row.get("strength"),
                        strength_overall_home=row.get("strength_overall_home"),
                        strength_overall_away=row.get("strength_overall_away"),
                        strength_attack_home=row.get("strength_attack_home"),
                        strength_attack_away=row.get("strength_attack_away"),
                        strength_defence_home=row.get("strength_defence_home"),
                        strength_defence_away=row.get("strength_defence_away"),
                    )
                )
            session.commit()
            logger.info(f"Persisted {len(df)} teams to database")
            return len(df)
        finally:
            session.close()

    def persist_fixtures(self) -> int:
        """Fetch fixtures from API and persist to the fixtures table."""
        df = self.get_fixtures_df()
        session = get_session()
        try:
            session.query(Fixture).delete()
            for _, row in df.iterrows():
                session.add(
                    Fixture(
                        id=row["id"],
                        event=row["event"],
                        team_h=row["team_h"],
                        team_a=row["team_a"],
                        team_h_name=row["team_h_name"],
                        team_a_name=row["team_a_name"],
                        team_h_score=row["team_h_score"],
                        team_a_score=row["team_a_score"],
                        finished=int(row["finished"]),
                        kickoff_time=row.get("kickoff_time"),
                    )
                )
            session.commit()
            logger.info(f"Persisted {len(df)} fixtures to database")
            return len(df)
        finally:
            session.close()
