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

import math
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
    """Dixon-Coles correction factor for low-scoring outcomes (scalar version)."""
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


def _neg_log_likelihood(
    params: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """Vectorised negative log-likelihood for the full dataset.

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

    # Expected goals — vectorised over all matches
    lam = attack[home_idx] * defence[away_idx] * home_adv  # (N,)
    mu = attack[away_idx] * defence[home_idx]               # (N,)

    # Poisson log-PMF for each match: k*log(rate) - rate - lgamma(k+1)
    hg = home_goals.astype(float)
    ag = away_goals.astype(float)
    lgamma_h = np.array([math.lgamma(int(g) + 1) for g in hg])
    lgamma_a = np.array([math.lgamma(int(g) + 1) for g in ag])
    log_pois = (
        hg * np.log(lam) - lam - lgamma_h
        + ag * np.log(mu) - mu - lgamma_a
    )

    # Tau correction — only 4 scorelines differ from 1.0
    log_tau = np.zeros(len(home_goals))
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    tau_vals = np.ones(len(home_goals))
    tau_vals[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau_vals[m01] = 1.0 + lam[m01] * rho
    tau_vals[m10] = 1.0 + mu[m10] * rho
    tau_vals[m11] = 1.0 - rho

    # Clamp to avoid log(<=0); heavily penalise invalid tau
    valid = tau_vals > 0
    log_tau[valid] = np.log(tau_vals[valid])
    log_tau[~valid] = -50.0

    total_ll = np.dot(weights, log_tau + log_pois)
    return -float(total_ll)


class DixonColesModel:
    """Dixon-Coles bivariate Poisson model for football match prediction."""

    def __init__(self, time_decay_xi: float = 0.0065) -> None:
        self.xi = time_decay_xi
        self.params: DixonColesParams | None = None
        self._teams: list[str] = []
        self._team_idx: dict[str, int] = {}

    def _fit_with_torch(
        self,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
        ref_idx: int = 0,
        max_iter: int = 500,
    ):
        """Fit Dixon-Coles parameters using PyTorch LBFGS (GPU-accelerated).

        Returns the scipy-style result namespace with attributes .x, .fun, .success.
        Falls back gracefully; caller handles exceptions.
        """
        import torch  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  PyTorch fitting on device: {device}")

        # Convert data to tensors (float64 for numerical precision)
        hg = torch.tensor(home_goals, dtype=torch.float64, device=device)
        ag = torch.tensor(away_goals, dtype=torch.float64, device=device)
        hi = torch.tensor(home_idx, dtype=torch.long, device=device)
        ai = torch.tensor(away_idx, dtype=torch.long, device=device)
        w = torch.tensor(weights, dtype=torch.float64, device=device)

        # Precompute lgamma once
        lgamma_h = torch.lgamma(hg + 1)
        lgamma_a = torch.lgamma(ag + 1)

        # Parameters: log-space for positivity (float64)
        log_attack = torch.zeros(n_teams, dtype=torch.float64, device=device, requires_grad=True)
        log_defence = torch.zeros(n_teams, dtype=torch.float64, device=device, requires_grad=True)
        log_home_adv = torch.tensor([np.log(1.25)], dtype=torch.float64, device=device, requires_grad=True)
        rho_param = torch.tensor([-0.05], dtype=torch.float64, device=device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [log_attack, log_defence, log_home_adv, rho_param],
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()

            # Enforce identifiability: ref team attack = 1 → log_attack[ref_idx] = 0
            with torch.no_grad():
                log_attack[ref_idx] = 0.0

            lam = torch.exp(log_attack[hi] + log_defence[ai] + log_home_adv)
            mu = torch.exp(log_attack[ai] + log_defence[hi])
            rho = rho_param[0]

            log_pois = (
                hg * torch.log(lam) - lam - lgamma_h
                + ag * torch.log(mu) - mu - lgamma_a
            )

            # Tau correction
            tau = torch.ones(len(hg), dtype=torch.float64, device=device)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau = torch.where(m00, 1.0 - lam * mu * rho, tau)
            tau = torch.where(m01, 1.0 + lam * rho, tau)
            tau = torch.where(m10, 1.0 + mu * rho, tau)
            tau = torch.where(m11, 1.0 - rho, tau)

            log_tau = torch.log(tau.clamp(min=1e-10))
            loss = -torch.dot(w, log_tau + log_pois)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            log_attack[ref_idx] = 0.0
            x_out = np.concatenate([
                log_attack.cpu().numpy(),
                log_defence.cpu().numpy(),
                log_home_adv.cpu().numpy(),
                rho_param.cpu().numpy(),
            ])
            loss_val = float(-np.dot(
                weights,
                np.log(np.ones(len(home_goals))),  # placeholder; recompute below
            ))

        # Re-evaluate final NLL on CPU for consistency
        final_nll = _neg_log_likelihood(
            x_out, home_goals, away_goals, home_idx, away_idx, weights, n_teams
        )

        class _Result:
            pass

        res = _Result()
        res.x = x_out
        res.fun = final_nll
        res.success = True
        return res

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

        # Try PyTorch GPU path first; fall back to scipy SLSQP
        result = None
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                logger.info("  CUDA available — fitting with PyTorch LBFGS on GPU")
                result = self._fit_with_torch(
                    home_goals, away_goals, home_idx, away_idx, weights, n_teams
                )
            else:
                logger.info("  CUDA not available — falling back to scipy SLSQP")
        except ImportError:
            logger.info("  PyTorch not installed — using scipy SLSQP")
        except Exception as exc:
            logger.warning(f"  PyTorch fitting failed ({exc}) — falling back to scipy")
            result = None

        if result is None:
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
        """Compute the full scoreline probability matrix (vectorised).

        Returns:
            (max_goals+1) x (max_goals+1) matrix where [i, j] = P(home=i, away=j).
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        lambda_, mu = self.predict_expected_goals(home_team, away_team)
        rho = self.params.rho

        goals = np.arange(max_goals + 1)
        # Outer product of independent Poisson PMFs
        matrix = np.outer(poisson.pmf(goals, lambda_), poisson.pmf(goals, mu))

        # Apply tau corrections only to the 4 affected cells
        matrix[0, 0] *= 1.0 - lambda_ * mu * rho
        matrix[0, 1] *= 1.0 + lambda_ * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho

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

    def register_team(
        self,
        name: str,
        attack: float | None = None,
        defence: float | None = None,
    ) -> None:
        """Add a team to the fitted model with given (or mean) parameters.

        Used for newly promoted teams that have no historical Premier League
        data and therefore were not included during fitting.  The team is
        appended with the mean attack and defence of all currently fitted teams
        if explicit values are not supplied.

        Args:
            name: Canonical team name (must match the normalised FPL name).
            attack: Attack rating. Defaults to mean of fitted teams.
            defence: Defence rating. Defaults to mean of fitted teams.
        """
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if name in self._team_idx:
            return  # already registered

        mean_atk = float(self.params.attack.mean())
        mean_def = float(self.params.defence.mean())
        atk = attack if attack is not None else mean_atk
        dfc = defence if defence is not None else mean_def

        self.params = DixonColesParams(
            teams=self.params.teams + [name],
            attack=np.append(self.params.attack, atk),
            defence=np.append(self.params.defence, dfc),
            home_adv=self.params.home_adv,
            rho=self.params.rho,
            log_likelihood=self.params.log_likelihood,
            converged=self.params.converged,
        )
        self._teams = self.params.teams
        self._team_idx[name] = len(self._teams) - 1
        logger.info(f"Registered unknown team '{name}' (atk={atk:.3f}, def={dfc:.3f})")
