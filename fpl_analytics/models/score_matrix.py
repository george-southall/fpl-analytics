"""Scoreline probability matrices and derived match predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fpl_analytics.models.dixon_coles import DixonColesModel
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchPrediction:
    """Full prediction for a single fixture."""

    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_cs_prob: float
    away_cs_prob: float
    score_matrix: np.ndarray
    most_likely_score: tuple[int, int]
    most_likely_score_prob: float


class ScoreMatrix:
    """Generate scoreline probability matrices and match predictions."""

    def __init__(self, model: DixonColesModel, max_goals: int = 7) -> None:
        self.model = model
        self.max_goals = max_goals

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        """Generate a full match prediction for a fixture."""
        matrix = self.model.predict_score_proba(
            home_team, away_team, self.max_goals
        )
        home_xg, away_xg = self.model.predict_expected_goals(home_team, away_team)

        # Result probabilities
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = matrix[i, j]
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

        # Clean sheet probabilities
        home_cs = float(matrix[:, 0].sum())  # away team scores 0
        away_cs = float(matrix[0, :].sum())  # home team scores 0

        # Most likely scoreline
        flat_idx = np.argmax(matrix)
        ml_home, ml_away = divmod(flat_idx, self.max_goals + 1)

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_xg=round(home_xg, 2),
            away_xg=round(away_xg, 2),
            home_win_prob=round(home_win, 4),
            draw_prob=round(draw, 4),
            away_win_prob=round(away_win, 4),
            home_cs_prob=round(home_cs, 4),
            away_cs_prob=round(away_cs, 4),
            score_matrix=matrix,
            most_likely_score=(int(ml_home), int(ml_away)),
            most_likely_score_prob=round(float(matrix[ml_home, ml_away]), 4),
        )

    def batch_predict(
        self, fixtures: list[tuple[str, str]]
    ) -> list[MatchPrediction]:
        """Predict multiple fixtures at once."""
        return [self.predict(h, a) for h, a in fixtures]

    @staticmethod
    def expected_goals_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
        """Compute expected goals from a score matrix."""
        n = matrix.shape[0]
        goals = np.arange(n)
        home_xg = float((matrix.sum(axis=1) * goals).sum())
        away_xg = float((matrix.sum(axis=0) * goals).sum())
        return home_xg, away_xg

    @staticmethod
    def btts_prob(matrix: np.ndarray) -> float:
        """Both teams to score probability."""
        return float(1.0 - matrix[:, 0].sum() - matrix[0, :].sum() + matrix[0, 0])

    @staticmethod
    def over_under_prob(matrix: np.ndarray, threshold: float = 2.5) -> tuple[float, float]:
        """Over/under probability for a total goals threshold."""
        n = matrix.shape[0]
        over = 0.0
        for i in range(n):
            for j in range(n):
                if i + j > threshold:
                    over += matrix[i, j]
        return float(over), float(1.0 - over)
