"""Shared test fixtures for FPL Analytics."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_players_df() -> pd.DataFrame:
    """Small sample of player data for testing."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "web_name": "Raya",
                "first_name": "David",
                "second_name": "Raya",
                "team_id": 1,
                "team_name": "Arsenal",
                "position_id": 1,
                "position": "GK",
                "now_cost": 60,
                "price": 6.0,
                "total_points": 140,
                "minutes": 2700,
                "goals_scored": 0,
                "assists": 0,
                "clean_sheets": 12,
                "selected_by_percent": 25.0,
                "transfers_in_event": 1000,
                "transfers_out_event": 500,
                "chance_of_playing_next_round": 100,
                "status": "a",
            },
            {
                "id": 233,
                "web_name": "Salah",
                "first_name": "Mohamed",
                "second_name": "Salah",
                "team_id": 11,
                "team_name": "Liverpool",
                "position_id": 3,
                "position": "MID",
                "now_cost": 133,
                "price": 13.3,
                "total_points": 220,
                "minutes": 2800,
                "goals_scored": 18,
                "assists": 12,
                "clean_sheets": 10,
                "selected_by_percent": 60.0,
                "transfers_in_event": 5000,
                "transfers_out_event": 2000,
                "chance_of_playing_next_round": 100,
                "status": "a",
            },
            {
                "id": 350,
                "web_name": "Isak",
                "first_name": "Alexander",
                "second_name": "Isak",
                "team_id": 14,
                "team_name": "Newcastle",
                "position_id": 4,
                "position": "FWD",
                "now_cost": 87,
                "price": 8.7,
                "total_points": 160,
                "minutes": 2400,
                "goals_scored": 15,
                "assists": 4,
                "clean_sheets": 0,
                "selected_by_percent": 35.0,
                "transfers_in_event": 3000,
                "transfers_out_event": 1500,
                "chance_of_playing_next_round": 100,
                "status": "a",
            },
        ]
    )


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Small synthetic results dataset for testing."""
    return pd.DataFrame(
        [
            {"date": "2025-01-01", "home_team": "Arsenal", "away_team": "Liverpool", "home_goals": 2, "away_goals": 1, "season": "2425"},
            {"date": "2025-01-08", "home_team": "Liverpool", "away_team": "Chelsea", "home_goals": 3, "away_goals": 0, "season": "2425"},
            {"date": "2025-01-15", "home_team": "Chelsea", "away_team": "Arsenal", "home_goals": 1, "away_goals": 1, "season": "2425"},
            {"date": "2025-01-22", "home_team": "Newcastle", "away_team": "Arsenal", "home_goals": 0, "away_goals": 2, "season": "2425"},
            {"date": "2025-01-29", "home_team": "Liverpool", "away_team": "Newcastle", "home_goals": 2, "away_goals": 2, "season": "2425"},
            {"date": "2025-02-05", "home_team": "Chelsea", "away_team": "Newcastle", "home_goals": 1, "away_goals": 0, "season": "2425"},
            {"date": "2025-02-12", "home_team": "Liverpool", "away_team": "Arsenal", "home_goals": 1, "away_goals": 2, "season": "2425"},
            {"date": "2025-02-19", "home_team": "Arsenal", "away_team": "Chelsea", "home_goals": 3, "away_goals": 1, "season": "2425"},
            {"date": "2025-02-26", "home_team": "Newcastle", "away_team": "Liverpool", "home_goals": 1, "away_goals": 3, "season": "2425"},
            {"date": "2025-03-05", "home_team": "Arsenal", "away_team": "Newcastle", "home_goals": 4, "away_goals": 0, "season": "2425"},
        ]
    )


@pytest.fixture
def sample_fixtures_df() -> pd.DataFrame:
    """Small sample of fixtures data for testing."""
    return pd.DataFrame(
        [
            {"id": 1, "event": 33, "team_h": 1, "team_a": 11, "team_h_name": "Arsenal", "team_a_name": "Liverpool", "team_h_score": None, "team_a_score": None, "finished": False, "kickoff_time": "2025-04-19T15:00:00Z"},
            {"id": 2, "event": 33, "team_h": 5, "team_a": 14, "team_h_name": "Chelsea", "team_a_name": "Newcastle", "team_h_score": None, "team_a_score": None, "finished": False, "kickoff_time": "2025-04-19T15:00:00Z"},
            {"id": 3, "event": 34, "team_h": 11, "team_a": 5, "team_h_name": "Liverpool", "team_a_name": "Chelsea", "team_h_score": None, "team_a_score": None, "finished": False, "kickoff_time": "2025-04-26T15:00:00Z"},
        ]
    )
