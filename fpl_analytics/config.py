"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FPL_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    db_path: Path = Path(__file__).resolve().parent.parent / "data" / "fpl_analytics.db"

    # FPL API
    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    fpl_team_id: int = 5823
    fpl_cache_ttl_hours: int = 6

    # football-data.co.uk
    football_data_base_url: str = "https://www.football-data.co.uk/mmz4281"
    seasons_to_fetch: int = 5

    # Dixon-Coles model
    dc_time_decay_xi: float = 0.0065
    dc_max_goals: int = 7

    # Optimiser
    squad_budget: float = 100.0
    max_players_per_club: int = 3

    # Projections
    projection_horizon_gws: int = 6
    trailing_minutes_window: int = 5

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
