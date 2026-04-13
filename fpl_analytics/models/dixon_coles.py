"""
Dixon-Coles Poisson regression model for football match prediction.

The Dixon-Coles model (1997) estimates team attacking and defensive strengths
by fitting a bivariate Poisson model to historical match results. A correction
term (rho) adjusts low-scoring outcomes (0-0, 1-0, 0-1, 1-1) which standard
Poisson slightly mispredicts.

Parameters estimated via maximum likelihood:
    - attack_i  (20 teams)
    - defence_i (20 teams)
    - home_advantage (scalar)
    - rho (correction term)

Reference: Dixon, M.J. and Coles, S.G. (1997). Modelling Association Football
Scores and Inefficiencies in the Football Betting Market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DixonColesParams:
    """Fitted model parameters."""

    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    log_likelihood: float
    converged: bool


def _tau(x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts the joint probability for scorelines 0-0, 1-0, 0-1, and 1-1
    where the independent Poisson assumption is weakest.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def _match_log_likelihood(
    home_goals: int,
    away_goals: int,
    lambda_: float,
    mu: float,
    rho: float,
    weight: float = 1.0,
) -> float:
    """Log-likelihood contribution of a single match."""
    tau = _tau(home_goals, away_goals, lambda_, mu, rho)
    if tau <= 0:
        return -50.0 * weight  # penalty for invalid tau

    log_lik = (
        np.log(tau)
        + poisson.logpmf(home_goals, lambda_)
        + poisson.logpmf(away_goals, mu)
    )
    return log_lik * weight


def _neg_log_likelihood(
    params: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """Negative log-likelihood for the full dataset.

    Parameter vector layout:
        [0 : n_teams]           = log(attack) for each team
        [n_teams : 2*n_teams]   = log(defence) for each team
        [2*n_teams]             = log(home_advantage)
        [2*n_teams + 1]         = rho
    """
    log_attack = params[:n_teams]
    log_defence = params[n_teams : 2 * n_teams]
    log_home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    attack = np.exp(log_attack)
    defence = np.exp(log_defence)
    home_adv = np.exp(log_home_adv)

    # Expected goals
    lambda_ = attack[home_idx] * defence[away_idx] * home_adv
    mu = attack[away_idx] * defence[home_idx]

    # Vectorised log-likelihood (tau correction applied per-match)
    total_ll = 0.0
    for i in range(len(home_goals)):
        total_ll += _match_log_likelihood(
            int(home_goals[i]),
            int(away_goals[i]),
            lambda_[i],
            mu[i],
            rho,
            weights[i],
        )

    return -total_ll


class DixonColesModel:
    """Dixon-Coles bivariate Poisson model for football match prediction."""

    def __init__(self, time_decay_xi: float = 0.0065) -> None:
        self.xi = time_decay_xi
        self.params: DixonColesParams | None = None
        self._teams: list[str] = []
        self._team_idx: dict[str, int] = {}

    def fit(self, results_df: pd.DataFrame) -> DixonColesParams:
        """Fit the model to historical results.

        Args:
            results_df: DataFrame with columns:
                home_team, away_team, home_goals, away_goals, date
                Optionally a 'weight' column (if absent, time decay is applied).

        Returns:
            Fitted DixonColesParams.
        """
        df = results_df.copy()

        # Apply time decay if no weight column present
        if "weight" not in df.columns:
            from fpl_analytics.ingestion.results_fetcher import apply_time_decay

            df = apply_time_decay(df, self.xi)

        # Build team index
        all_teams = sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        )
        self._teams = all_teams
        self._team_idx = {team: i for i, team in enumerate(all_teams)}
        n_teams = len(all_teams)

        logger.info(f"Fitting Dixon-Coles model on {len(df)} matches, {n_teams} teams")

        # Map teams to indices
        home_idx = df["home_team"].map(self._team_idx).values.astype(int)
        away_idx = df["away_team"].map(self._team_idx).values.astype(int)
        home_goals = df["home_goals"].values.astype(int)
        away_goals = df["away_goals"].values.astype(int)
        weights = df["weight"].values.astype(float)

        # Initial parameters: all attacks/defences = 1.0, home_adv = 1.25, rho = 0
        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams] = np.log(1.25)  # home advantage
        x0[2 * n_teams + 1] = -0.05  # rho initial guess

        # Constraint: fix reference team attack = 1.0 (log = 0) for identifiability
        # We use the first team alphabetically (typically Arsenal)
        ref_idx = 0

        def constraint_fn(params):
            return params[ref_idx]  # log(attack_ref) = 0 → attack_ref = 1.0

        constraints = [{"type": "eq", "fun": constraint_fn}]

        # Bounds: rho typically in [-1, 1], log params unbounded
        bounds = [(None, None)] * (2 * n_teams + 1) + [(-1.5, 1.5)]

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(home_goals, away_goals, home_idx, away_idx, weights, n_teams),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        attack = np.exp(result.x[:n_teams])
        defence = np.exp(result.x[n_teams : 2 * n_teams])
        home_adv = np.exp(result.x[2 * n_teams])
        rho = result.x[2 * n_teams + 1]

        self.params = DixonColesParams(
            teams=all_teams,
            attack=attack,
            defence=defence,
            home_adv=home_adv,
            rho=rho,
            log_likelihood=-result.fun,
            converged=result.success,
        )

        logger.info(f"  Converged: {result.success}")
        logger.info(f"  Log-likelihood: {-result.fun:.2f}")
        logger.info(f"  Home advantage: {home_adv:.4f}")
        logger.info(f"  Rho: {rho:.4f}")

        if not result.success:
            logger.warning(f"  Optimisation message: {result.message}")

        return self.params

    def predict_expected_goals(
        self, home_team: str, away_team: str
    ) -> tuple[float, float]:
        """Predict expected goals for a fixture.

        Returns:
            (lambda_home, mu_away) — expected goals for home and away teams.
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        h_idx = self._team_idx[home_team]
        a_idx = self._team_idx[away_team]

        lambda_ = (
            self.params.attack[h_idx]
            * self.params.defence[a_idx]
            * self.params.home_adv
        )
        mu = self.params.attack[a_idx] * self.params.defence[h_idx]

        return float(lambda_), float(mu)

    def predict_score_proba(
        self, home_team: str, away_team: str, max_goals: int = 7
    ) -> np.ndarray:
        """Compute the full scoreline probability matrix.

        Returns:
            (max_goals+1) x (max_goals+1) matrix where [i, j] = P(home=i, away=j).
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        lambda_, mu = self.predict_expected_goals(home_team, away_team)
        rho = self.params.rho

        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = _tau(i, j, lambda_, mu, rho)
                matrix[i, j] = (
                    tau
                    * poisson.pmf(i, lambda_)
                    * poisson.pmf(j, mu)
                )

        return matrix

    def get_team_attack(self, team: str) -> float:
        """Get attack rating for a team."""
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return float(self.params.attack[self._team_idx[team]])

    def get_team_defence(self, team: str) -> float:
        """Get defence rating for a team."""
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return float(self.params.defence[self._team_idx[team]])
