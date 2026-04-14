"""Extract, store, and compare team attack/defence ratings."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl_analytics.config import settings
from fpl_analytics.models.dixon_coles import DixonColesModel, DixonColesParams
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


class TeamStrengths:
    """Compute and manage team strength ratings from the Dixon-Coles model."""

    def __init__(self, model: DixonColesModel | None = None) -> None:
        self.model = model or DixonColesModel(time_decay_xi=settings.dc_time_decay_xi)
        self._params: DixonColesParams | None = None

    def fit(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Fit the Dixon-Coles model and return a team strengths DataFrame.

        Returns:
            DataFrame with columns: team, attack, defence, overall, last_updated
        """
        self._params = self.model.fit(results_df)
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fitted parameters to a clean DataFrame.

        Always reads from self.model.params so that teams added via
        register_team() after fitting are included.
        """
        # Always use the live params (not the cached _params) so any teams
        # injected via register_team() after fitting are included.
        params = self.model.params
        if params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for i, team in enumerate(params.teams):
            att = float(params.attack[i])
            def_ = float(params.defence[i])
            rows.append(
                {
                    "team": team,
                    "attack": round(att, 4),
                    "defence": round(def_, 4),
                    "overall": round(att / def_, 4),  # higher = stronger overall
                    "last_updated": now,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("overall", ascending=False).reset_index(drop=True)
        return df

    def save(self, path: Path | None = None) -> Path:
        """Save team strengths to CSV."""
        path = path or settings.data_dir / "processed" / "team_strengths.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved team strengths to {path}")
        return path

    def load(self, path: Path | None = None) -> pd.DataFrame:
        """Load team strengths from CSV."""
        path = path or settings.data_dir / "processed" / "team_strengths.csv"
        return pd.read_csv(path)

    def compare(self, other_df: pd.DataFrame) -> pd.DataFrame:
        """Compare our ratings with another source (e.g. Solio).

        Args:
            other_df: DataFrame with columns: team, attack, defence

        Returns:
            Merged DataFrame showing both sets of ratings and the difference.
        """
        ours = self.to_dataframe()[["team", "attack", "defence"]]
        ours = ours.rename(columns={"attack": "our_attack", "defence": "our_defence"})

        other = other_df[["team", "attack", "defence"]].copy()
        other = other.rename(columns={"attack": "other_attack", "defence": "other_defence"})

        merged = ours.merge(other, on="team", how="outer")
        merged["attack_diff"] = merged["our_attack"] - merged["other_attack"]
        merged["defence_diff"] = merged["our_defence"] - merged["other_defence"]

        return merged

    @property
    def home_advantage(self) -> float:
        """The fitted home advantage parameter."""
        if self._params is None:
            raise RuntimeError("Model not fitted.")
        return self._params.home_adv

    @property
    def rho(self) -> float:
        """The fitted rho correction parameter."""
        if self._params is None:
            raise RuntimeError("Model not fitted.")
        return self._params.rho

    @property
    def log_likelihood(self) -> float:
        """The log-likelihood of the fitted model."""
        if self._params is None:
            raise RuntimeError("Model not fitted.")
        return self._params.log_likelihood
