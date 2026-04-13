"""Tests for the data validator."""

import pandas as pd

from fpl_analytics.ingestion.data_validator import DataValidator


class TestDataValidator:
    def test_validate_players_passes_with_good_data(self, sample_players_df):
        validator = DataValidator()
        validator.validate_players(sample_players_df)
        # Small sample won't pass count check but should pass others
        results = {r.check: r.passed for r in validator.results}
        assert results["players_no_duplicates"]
        assert results["players_valid_positions"]
        assert results["players_price_range"]

    def test_validate_players_detects_duplicate_ids(self, sample_players_df):
        df = pd.concat([sample_players_df, sample_players_df.iloc[:1]], ignore_index=True)
        validator = DataValidator()
        validator.validate_players(df)
        results = {r.check: r.passed for r in validator.results}
        assert not results["players_no_duplicates"]

    def test_validate_results_passes_with_good_data(self, sample_results_df):
        validator = DataValidator()
        validator.validate_results(sample_results_df)
        results = {r.check: r.passed for r in validator.results}
        assert results["results_no_missing_goals"]
        assert results["results_goal_range"]
        assert results["results_no_duplicates"]

    def test_validate_results_detects_duplicates(self, sample_results_df):
        df = pd.concat([sample_results_df, sample_results_df.iloc[:1]], ignore_index=True)
        validator = DataValidator()
        validator.validate_results(df)
        results = {r.check: r.passed for r in validator.results}
        assert not results["results_no_duplicates"]

    def test_validate_fixtures(self, sample_fixtures_df):
        validator = DataValidator()
        validator.validate_fixtures(sample_fixtures_df)
        # Small sample won't pass count check, but should not error
        assert len(validator.results) > 0

    def test_validate_team_name_consistency(self):
        validator = DataValidator()
        fpl_teams = ["Arsenal", "Liverpool", "Chelsea"]
        results_teams = ["Arsenal", "Liverpool", "Chelsea", "Leeds"]
        validator.validate_team_name_consistency(fpl_teams, results_teams)
        results = {r.check: r.passed for r in validator.results}
        assert results["team_names_fpl_in_results"] is True

    def test_validate_team_name_consistency_detects_missing(self):
        validator = DataValidator()
        fpl_teams = ["Arsenal", "Liverpool", "Chelsea"]
        results_teams = ["Arsenal", "Liverpool"]  # Chelsea missing
        validator.validate_team_name_consistency(fpl_teams, results_teams)
        results = {r.check: r.passed for r in validator.results}
        assert results["team_names_fpl_in_results"] is False

    def test_summary(self, sample_players_df):
        validator = DataValidator()
        validator.validate_players(sample_players_df)
        summary = validator.summary()
        assert "total" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert summary["total"] == len(validator.results)
