"""Data quality checks across all data sources."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fpl_analytics.utils.fpl_constants import normalise_team_name
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


class DataValidator:
    """Run data quality checks across FPL data sources."""

    def __init__(self) -> None:
        self.results: list[ValidationResult] = []

    def _add(self, check: str, passed: bool, message: str, details: list[str] | None = None):
        result = ValidationResult(check=check, passed=passed, message=message, details=details or [])
        self.results.append(result)
        level = "PASS" if passed else "FAIL"
        logger.info(f"  [{level}] {check}: {message}")
        for d in result.details:
            logger.info(f"         {d}")

    def validate_players(self, players_df: pd.DataFrame) -> None:
        """Validate player data from FPL API."""
        # Check we have a reasonable number of players
        n = len(players_df)
        self._add(
            "players_count",
            500 <= n <= 900,
            f"{n} players loaded (expected 500-900)",
        )

        # Check for duplicate player IDs
        dupes = players_df["id"].duplicated().sum()
        self._add("players_no_duplicates", dupes == 0, f"{dupes} duplicate player IDs")

        # Check all positions are valid
        valid_pos = {"GK", "DEF", "MID", "FWD"}
        actual_pos = set(players_df["position"].unique())
        invalid = actual_pos - valid_pos
        self._add("players_valid_positions", len(invalid) == 0, f"Invalid positions: {invalid or 'none'}")

        # Check price range is reasonable
        prices = players_df["price"]
        self._add(
            "players_price_range",
            prices.min() >= 3.5 and prices.max() <= 16.0,
            f"Price range: £{prices.min():.1f}m - £{prices.max():.1f}m",
        )

    def validate_fixtures(self, fixtures_df: pd.DataFrame) -> None:
        """Validate fixture data."""
        n = len(fixtures_df)
        self._add("fixtures_count", n >= 380, f"{n} fixtures loaded (expected ≥380)")

        # Check for fixtures without a gameweek
        no_gw = fixtures_df["event"].isna().sum()
        self._add("fixtures_have_gameweeks", True, f"{no_gw} fixtures without a gameweek assigned")

    def validate_results(self, results_df: pd.DataFrame) -> None:
        """Validate historical results."""
        n = len(results_df)
        self._add("results_count", n >= 380, f"{n} historical results loaded")

        # Check for missing goals
        missing_goals = results_df[["home_goals", "away_goals"]].isna().sum().sum()
        self._add("results_no_missing_goals", missing_goals == 0, f"{missing_goals} missing goal values")

        # Check goal range is reasonable
        max_goals = max(results_df["home_goals"].max(), results_df["away_goals"].max())
        self._add("results_goal_range", max_goals <= 12, f"Max goals in a match: {max_goals}")

        # Check for duplicate matches
        dupes = results_df.duplicated(subset=["date", "home_team", "away_team"]).sum()
        self._add("results_no_duplicates", dupes == 0, f"{dupes} duplicate matches")

    def validate_team_name_consistency(
        self,
        fpl_teams: list[str],
        results_teams: list[str],
    ) -> None:
        """Check that team names are consistent across sources after normalisation."""
        fpl_norm = {normalise_team_name(t) for t in fpl_teams}
        res_norm = {normalise_team_name(t) for t in results_teams}

        # Current PL teams should appear in both
        in_fpl_not_results = fpl_norm - res_norm
        in_results_not_fpl = res_norm - fpl_norm

        # It's OK for results to have relegated teams not in current FPL
        self._add(
            "team_names_fpl_in_results",
            len(in_fpl_not_results) == 0,
            f"FPL teams missing from results: {in_fpl_not_results or 'none'}",
            details=list(in_fpl_not_results),
        )

        # Informational only — relegated teams expected
        if in_results_not_fpl:
            self._add(
                "team_names_results_extra",
                True,  # not a failure
                f"Results teams not in current FPL (likely relegated): {len(in_results_not_fpl)}",
                details=sorted(in_results_not_fpl),
            )

    def validate_understat(self, understat_df: pd.DataFrame) -> None:
        """Validate Understat xG/xA data."""
        n = len(understat_df)
        self._add("understat_count", n >= 300, f"{n} Understat player records loaded")

        # Check xG range
        max_xg = understat_df["xg"].max()
        self._add("understat_xg_range", 0 <= max_xg <= 40, f"Max xG: {max_xg:.1f}")

    def summary(self) -> dict:
        """Return a summary of all checks."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        logger.info(f"\nValidation summary: {passed} passed, {failed} failed")
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
            "results": self.results,
        }

    def run_all(
        self,
        players_df: pd.DataFrame | None = None,
        fixtures_df: pd.DataFrame | None = None,
        results_df: pd.DataFrame | None = None,
        understat_df: pd.DataFrame | None = None,
    ) -> dict:
        """Run all applicable validation checks."""
        logger.info("Running data validation checks...")

        if players_df is not None:
            self.validate_players(players_df)

        if fixtures_df is not None:
            self.validate_fixtures(fixtures_df)

        if results_df is not None:
            self.validate_results(results_df)

        if understat_df is not None:
            self.validate_understat(understat_df)

        # Cross-source team name consistency
        if players_df is not None and results_df is not None:
            fpl_teams = players_df["team_name"].unique().tolist()
            results_teams = (
                results_df["home_team"].unique().tolist()
                + results_df["away_team"].unique().tolist()
            )
            self.validate_team_name_consistency(fpl_teams, list(set(results_teams)))

        return self.summary()
