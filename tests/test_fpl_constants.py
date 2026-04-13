"""Tests for FPL constants and utility functions."""

from fpl_analytics.utils.fpl_constants import (
    CS_POINTS,
    GOAL_POINTS,
    POINTS,
    POSITION_MAP,
    SQUAD_COMPOSITION,
    VALID_FORMATIONS,
    normalise_team_name,
)


class TestFPLConstants:
    def test_position_map_has_four_positions(self):
        assert len(POSITION_MAP) == 4
        assert POSITION_MAP[1] == "GK"
        assert POSITION_MAP[4] == "FWD"

    def test_goal_points_by_position(self):
        assert GOAL_POINTS["GK"] == 10
        assert GOAL_POINTS["DEF"] == 6
        assert GOAL_POINTS["MID"] == 5
        assert GOAL_POINTS["FWD"] == 4

    def test_cs_points_by_position(self):
        assert CS_POINTS["GK"] == 4
        assert CS_POINTS["DEF"] == 4
        assert CS_POINTS["MID"] == 1
        assert CS_POINTS["FWD"] == 0

    def test_squad_composition_totals_15(self):
        assert sum(SQUAD_COMPOSITION.values()) == 15

    def test_all_formations_have_11_players(self):
        for formation in VALID_FORMATIONS:
            assert sum(formation) == 10  # outfield only; GK=1 always
            d, m, f = formation
            assert 3 <= d <= 5
            assert 2 <= m <= 5
            assert 1 <= f <= 3

    def test_points_dict_has_expected_keys(self):
        assert "goal_mid" in POINTS
        assert "assist" in POINTS
        assert "clean_sheet_def" in POINTS
        assert "yellow_card" in POINTS


class TestNormaliseTeamName:
    def test_direct_match(self):
        assert normalise_team_name("Arsenal") == "Arsenal"

    def test_fpl_short_names(self):
        assert normalise_team_name("Man City") == "Manchester City"
        assert normalise_team_name("Man Utd") == "Manchester United"
        assert normalise_team_name("Spurs") == "Tottenham"
        assert normalise_team_name("Nott'm Forest") == "Nottingham Forest"

    def test_football_data_names(self):
        assert normalise_team_name("Manchester City") == "Manchester City"
        assert normalise_team_name("Wolverhampton Wanderers") == "Wolverhampton"
        assert normalise_team_name("Brighton and Hove Albion") == "Brighton"

    def test_unknown_name_returned_as_is(self):
        assert normalise_team_name("Unknown FC") == "Unknown FC"
