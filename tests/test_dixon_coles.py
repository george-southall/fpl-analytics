"""Tests for the Dixon-Coles model.

Uses a synthetic dataset with known team strengths to verify
the model can recover parameters within tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from fpl_analytics.models.dixon_coles import DixonColesModel, _tau
from fpl_analytics.models.score_matrix import ScoreMatrix
from fpl_analytics.models.team_strengths import TeamStrengths


def _generate_synthetic_results(
    n_matches_per_pair: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate synthetic match results from known parameters.

    Creates a 4-team league with known attack/defence ratings and
    simulates matches using Poisson draws.
    """
    rng = np.random.default_rng(seed)

    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    true_params = {
        "attack": {"TeamA": 1.0, "TeamB": 1.4, "TeamC": 0.7, "TeamD": 1.1},
        "defence": {"TeamA": 0.9, "TeamB": 1.0, "TeamC": 1.3, "TeamD": 1.1},
        "home_adv": 1.2,
    }

    rows = []
    match_num = 0
    for _ in range(n_matches_per_pair):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lambda_ = (
                    true_params["attack"][h]
                    * true_params["defence"][a]
                    * true_params["home_adv"]
                )
                mu = true_params["attack"][a] * true_params["defence"][h]
                hg = rng.poisson(lambda_)
                ag = rng.poisson(mu)
                rows.append(
                    {
                        "date": f"2025-01-{1 + match_num % 28:02d}",
                        "home_team": h,
                        "away_team": a,
                        "home_goals": hg,
                        "away_goals": ag,
                        "season": "2425",
                        "weight": 1.0,  # equal weights for synthetic data
                    }
                )
                match_num += 1

    return pd.DataFrame(rows), true_params


class TestTauCorrection:
    """Test the Dixon-Coles tau correction factor."""

    def test_tau_00(self):
        tau = _tau(0, 0, 1.5, 1.0, -0.1)
        assert tau == pytest.approx(1.0 - 1.5 * 1.0 * (-0.1))

    def test_tau_01(self):
        tau = _tau(0, 1, 1.5, 1.0, -0.1)
        assert tau == pytest.approx(1.0 + 1.5 * (-0.1))

    def test_tau_10(self):
        tau = _tau(1, 0, 1.5, 1.0, -0.1)
        assert tau == pytest.approx(1.0 + 1.0 * (-0.1))

    def test_tau_11(self):
        tau = _tau(1, 1, 1.5, 1.0, -0.1)
        assert tau == pytest.approx(1.0 - (-0.1))

    def test_tau_other_scores_are_1(self):
        for h, a in [(2, 0), (0, 2), (2, 1), (3, 3), (5, 0)]:
            assert _tau(h, a, 1.5, 1.0, -0.1) == 1.0


class TestDixonColesModel:
    """Test the full model fitting and prediction."""

    @pytest.fixture
    def synthetic_data(self):
        return _generate_synthetic_results(n_matches_per_pair=12, seed=42)

    @pytest.fixture
    def fitted_model(self, synthetic_data):
        df, _ = synthetic_data
        model = DixonColesModel(time_decay_xi=0.0)  # no decay for synthetic
        model.fit(df)
        return model

    def test_model_converges(self, synthetic_data):
        df, _ = synthetic_data
        model = DixonColesModel(time_decay_xi=0.0)
        params = model.fit(df)
        assert params.converged

    def test_reference_team_attack_is_one(self, fitted_model):
        """First team alphabetically should have attack = 1.0 (reference)."""
        ref_team = fitted_model.params.teams[0]  # "TeamA"
        assert fitted_model.get_team_attack(ref_team) == pytest.approx(1.0, abs=0.01)

    def test_recovers_relative_attack_order(self, synthetic_data, fitted_model):
        """TeamB should have the highest attack rating."""
        _, true_params = synthetic_data
        attacks = {
            t: fitted_model.get_team_attack(t) for t in fitted_model.params.teams
        }
        strongest_attacker = max(attacks, key=attacks.get)
        assert strongest_attacker == "TeamB"

    def test_recovers_relative_defence_order(self, synthetic_data, fitted_model):
        """TeamC should have the highest (worst) defence rating."""
        _, true_params = synthetic_data
        defences = {
            t: fitted_model.get_team_defence(t) for t in fitted_model.params.teams
        }
        weakest_defence = max(defences, key=defences.get)
        assert weakest_defence == "TeamC"

    def test_home_advantage_positive(self, fitted_model):
        """Home advantage should be > 1."""
        assert fitted_model.params.home_adv > 1.0

    def test_home_advantage_reasonable(self, fitted_model):
        """Home advantage should be within a reasonable range."""
        assert 1.0 < fitted_model.params.home_adv < 2.0

    def test_predict_expected_goals_positive(self, fitted_model):
        lambda_, mu = fitted_model.predict_expected_goals("TeamA", "TeamB")
        assert lambda_ > 0
        assert mu > 0

    def test_score_matrix_sums_to_one(self, fitted_model):
        matrix = fitted_model.predict_score_proba("TeamA", "TeamB")
        assert matrix.sum() == pytest.approx(1.0, abs=0.02)

    def test_score_matrix_shape(self, fitted_model):
        matrix = fitted_model.predict_score_proba("TeamA", "TeamB", max_goals=6)
        assert matrix.shape == (7, 7)

    def test_score_matrix_non_negative(self, fitted_model):
        matrix = fitted_model.predict_score_proba("TeamA", "TeamB")
        assert (matrix >= 0).all()


class TestScoreMatrix:
    """Test the ScoreMatrix prediction wrapper."""

    @pytest.fixture
    def predictor(self):
        df, _ = _generate_synthetic_results(n_matches_per_pair=12, seed=42)
        model = DixonColesModel(time_decay_xi=0.0)
        model.fit(df)
        return ScoreMatrix(model)

    def test_prediction_probabilities_sum_to_one(self, predictor):
        pred = predictor.predict("TeamA", "TeamB")
        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert total == pytest.approx(1.0, abs=0.02)

    def test_clean_sheet_probs_reasonable(self, predictor):
        pred = predictor.predict("TeamA", "TeamB")
        assert 0 < pred.home_cs_prob < 1
        assert 0 < pred.away_cs_prob < 1

    def test_most_likely_score_exists(self, predictor):
        pred = predictor.predict("TeamA", "TeamB")
        h, a = pred.most_likely_score
        assert h >= 0
        assert a >= 0
        assert pred.most_likely_score_prob > 0

    def test_batch_predict(self, predictor):
        fixtures = [("TeamA", "TeamB"), ("TeamC", "TeamD")]
        preds = predictor.batch_predict(fixtures)
        assert len(preds) == 2

    def test_btts_probability(self, predictor):
        pred = predictor.predict("TeamA", "TeamB")
        btts = ScoreMatrix.btts_prob(pred.score_matrix)
        assert 0 < btts < 1

    def test_over_under(self, predictor):
        pred = predictor.predict("TeamA", "TeamB")
        over, under = ScoreMatrix.over_under_prob(pred.score_matrix, 2.5)
        assert over + under == pytest.approx(1.0, abs=0.02)


class TestTeamStrengths:
    """Test the TeamStrengths wrapper."""

    def test_fit_returns_dataframe(self):
        df, _ = _generate_synthetic_results(n_matches_per_pair=10, seed=42)
        ts = TeamStrengths(DixonColesModel(time_decay_xi=0.0))
        result = ts.fit(df)
        assert isinstance(result, pd.DataFrame)
        assert "team" in result.columns
        assert "attack" in result.columns
        assert "defence" in result.columns
        assert "overall" in result.columns
        assert len(result) == 4

    def test_save_and_load(self, tmp_path):
        df, _ = _generate_synthetic_results(n_matches_per_pair=10, seed=42)
        ts = TeamStrengths(DixonColesModel(time_decay_xi=0.0))
        ts.fit(df)
        path = ts.save(tmp_path / "strengths.csv")
        loaded = ts.load(path)
        assert len(loaded) == 4
        assert "attack" in loaded.columns

    def test_compare(self):
        df, _ = _generate_synthetic_results(n_matches_per_pair=10, seed=42)
        ts = TeamStrengths(DixonColesModel(time_decay_xi=0.0))
        ts.fit(df)

        other = pd.DataFrame(
            {
                "team": ["TeamA", "TeamB", "TeamC", "TeamD"],
                "attack": [1.0, 1.3, 0.8, 1.0],
                "defence": [0.9, 1.0, 1.2, 1.0],
            }
        )
        comparison = ts.compare(other)
        assert "attack_diff" in comparison.columns
        assert "defence_diff" in comparison.columns
        assert len(comparison) == 4
