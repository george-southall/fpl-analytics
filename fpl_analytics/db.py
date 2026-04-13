"""SQLAlchemy database setup and models for caching FPL data."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from fpl_analytics.config import settings


class Base(DeclarativeBase):
    pass


class CacheEntry(Base):
    """Generic cache table for storing API responses with TTL."""

    __tablename__ = "cache"

    key = Column(String, primary_key=True)
    data = Column(Text, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Player(Base):
    """FPL player master data from bootstrap-static."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    web_name = Column(String, nullable=False)
    first_name = Column(String)
    second_name = Column(String)
    team_id = Column(Integer, nullable=False)
    team_name = Column(String)
    position_id = Column(Integer, nullable=False)
    position = Column(String)
    now_cost = Column(Integer)  # price × 10
    total_points = Column(Integer)
    minutes = Column(Integer)
    goals_scored = Column(Integer)
    assists = Column(Integer)
    clean_sheets = Column(Integer)
    selected_by_percent = Column(Float)
    transfers_in_event = Column(Integer)
    transfers_out_event = Column(Integer)
    chance_of_playing_next_round = Column(Integer)
    status = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Team(Base):
    """FPL team data from bootstrap-static."""

    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    short_name = Column(String)
    strength = Column(Integer)
    strength_overall_home = Column(Integer)
    strength_overall_away = Column(Integer)
    strength_attack_home = Column(Integer)
    strength_attack_away = Column(Integer)
    strength_defence_home = Column(Integer)
    strength_defence_away = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Fixture(Base):
    """FPL fixture data."""

    __tablename__ = "fixtures"

    id = Column(Integer, primary_key=True)
    event = Column(Integer)  # gameweek number
    team_h = Column(Integer, nullable=False)
    team_a = Column(Integer, nullable=False)
    team_h_name = Column(String)
    team_a_name = Column(String)
    team_h_score = Column(Integer)
    team_a_score = Column(Integer)
    finished = Column(Integer)  # boolean as int
    kickoff_time = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)


class HistoricalResult(Base):
    """Historical Premier League match results from football-data.co.uk."""

    __tablename__ = "historical_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(String, nullable=False)
    date = Column(String, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_goals = Column(Integer, nullable=False)
    away_goals = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)


class UnderstatPlayer(Base):
    """Per-match xG/xA data from Understat."""

    __tablename__ = "understat_players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String, nullable=False)
    team = Column(String)
    season = Column(String)
    matches = Column(Integer)
    goals = Column(Integer)
    xg = Column(Float)
    assists = Column(Integer)
    xa = Column(Float)
    npg = Column(Integer)
    npxg = Column(Float)
    xg_per_90 = Column(Float)
    xa_per_90 = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


# Engine and session factory
engine = create_engine(settings.db_url, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables if they don't exist."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()
