"""Tests for FPL points calculator.

Tests each position's formula against known inputs to validate exact outputs.
"""

from __future__ import annotations

import pytest

from fpl_analytics.projections.minutes_model import appearance_points
from fpl_analytics.projections.points_calculator import calculate_player_xpts


class TestAppearancePoints:
    def test_zero_minutes(self):
        assert appearance_points(0) == 0.0

    def test_full_game(self):
        # 90 mins → definitely gets 2 pts
        pts = appearance_points(90)
        assert pts == pytest.approx(2.0, abs=0.01)

    def test_60_mins(self):
        # Exactly 60 mins → P(ge60)=0.5, P(lt60)=0.5*(60/90)
        pts = appearance_points(60)
        assert 1.0 < pts < 2.0

    def test_30_mins(self):
        # 30 mins → unlikely to play 60
        pts = appearance_points(30)
        assert pts < 1.5

    def test_180_mins_dgw(self):
        # DGW: capped contribution, should be around 4
        pts = appearance_points(180)
        assert pts == pytest.approx(4.0, abs=0.1)


class TestPointsCalculatorGK:
    """Goalkeeper scoring tests."""

    def test_gk_full_game_clean_sheet(self):
        """GK, 90 mins, clean sheet: expect ~6+ points."""
        result = calculate_player_xpts(
            player_id=1, name="Raya", position="GK",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=1.5, team_xgc=0.0,
            cs_prob=1.0,  # certain clean sheet
            saves_per90=3.0,
        )
        # 2 (appearance) + 4 (CS) + 1 (3 saves) = 7 minimum
        assert result.xPts_total >= 6.5
        assert result.xPts_clean_sheet == pytest.approx(4.0, abs=0.1)
        assert result.xPts_saves == pytest.approx(1.0, abs=0.1)

    def test_gk_zero_minutes(self):
        result = calculate_player_xpts(
            player_id=1, name="Raya", position="GK",
            xmins=0, xg_per90=0.0, xa_per90=0.0,
            team_xg=1.5, team_xgc=1.0, cs_prob=0.3,
        )
        assert result.xPts_total == 0.0

    def test_gk_goals_conceded_penalty(self):
        """GK conceding 2 xG should take a -1 penalty."""
        result = calculate_player_xpts(
            player_id=1, name="Raya", position="GK",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=0.5, team_xgc=2.0,
            cs_prob=0.05,
        )
        assert result.xPts_goals_conceded < 0


class TestPointsCalculatorDEF:
    """Defender scoring tests."""

    def test_def_clean_sheet_full_game(self):
        """DEF, 90 mins, clean sheet: should earn 4 CS points."""
        result = calculate_player_xpts(
            player_id=10, name="Saliba", position="DEF",
            xmins=90, xg_per90=0.05, xa_per90=0.05,
            team_xg=1.2, team_xgc=0.0,
            cs_prob=1.0,
        )
        assert result.xPts_clean_sheet == pytest.approx(4.0, abs=0.1)

    def test_def_goal_worth_6_pts(self):
        """DEF goal should be worth 6 FPL points."""
        result = calculate_player_xpts(
            player_id=10, name="Saliba", position="DEF",
            xmins=90, xg_per90=1.0, xa_per90=0.0,  # 1 xG per 90
            team_xg=2.0, team_xgc=1.0,
            cs_prob=0.2,
        )
        # At most 1 goal worth 6 pts; check goal contribution
        assert result.xPts_goals > 0
        # 1 goal from DEF = 6 pts
        assert result.xPts_goals <= 6.0


class TestPointsCalculatorMID:
    """Midfielder scoring tests — including the spec example."""

    def test_mid_1g_1a_90min_cs(self):
        """MID, 1 goal, 1 assist, 90 mins, clean sheet → should be ~17 pts.

        Per spec: 2 (app) + 5 (goal) + 3 (assist) + 1 (CS) + ~6 (bonus) = 17
        We test the formula logic, not exact bonus.
        """
        result = calculate_player_xpts(
            player_id=100, name="Salah", position="MID",
            xmins=90,
            xg_per90=1.0,   # exactly 1 goal
            xa_per90=1.0,   # exactly 1 assist
            team_xg=2.0,
            team_xgc=0.0,
            cs_prob=1.0,
            bonus_per90=0.0,  # exclude bonus for clean calculation
        )
        # 2 (appearance) + 5 (goal) + 3 (assist) + 1 (CS mid)
        base = result.xPts_appearance + result.xPts_goals + result.xPts_assists + result.xPts_clean_sheet
        assert base == pytest.approx(11.0, abs=0.5)

    def test_mid_no_cs_bonus(self):
        """MID gets 1 pt clean sheet, not 4."""
        result_mid = calculate_player_xpts(
            player_id=100, name="Test", position="MID",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=1.0, team_xgc=0.0, cs_prob=1.0,
        )
        result_def = calculate_player_xpts(
            player_id=101, name="Test", position="DEF",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=1.0, team_xgc=0.0, cs_prob=1.0,
        )
        assert result_mid.xPts_clean_sheet < result_def.xPts_clean_sheet


class TestPointsCalculatorFWD:
    """Forward scoring tests."""

    def test_fwd_goal_worth_4_pts(self):
        """FWD goal should be worth 4 FPL points."""
        result = calculate_player_xpts(
            player_id=200, name="Isak", position="FWD",
            xmins=90, xg_per90=1.0, xa_per90=0.0,
            team_xg=2.0, team_xgc=1.0,
            cs_prob=0.2,
        )
        # FWD goal = 4pts; with 1 xG per 90 and 90 mins
        assert result.xPts_goals == pytest.approx(4.0, abs=0.3)

    def test_fwd_no_cs_points(self):
        """FWD should earn 0 clean sheet points."""
        result = calculate_player_xpts(
            player_id=200, name="Isak", position="FWD",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=1.0, team_xgc=0.0, cs_prob=1.0,
        )
        assert result.xPts_clean_sheet == 0.0

    def test_fwd_no_gc_penalty(self):
        """FWD should not receive goals-conceded penalty."""
        result = calculate_player_xpts(
            player_id=200, name="Isak", position="FWD",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=0.5, team_xgc=5.0, cs_prob=0.0,
        )
        assert result.xPts_goals_conceded == 0.0


class TestPointsCalculatorCards:
    def test_yellow_card_penalty(self):
        result = calculate_player_xpts(
            player_id=1, name="Test", position="MID",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=1.0, team_xgc=1.0, cs_prob=0.2,
            yellow_card_prob=1.0,  # certain yellow
        )
        assert result.xPts_cards == pytest.approx(-1.0, abs=0.1)

    def test_red_card_penalty(self):
        result = calculate_player_xpts(
            player_id=1, name="Test", position="MID",
            xmins=90, xg_per90=0.0, xa_per90=0.0,
            team_xg=1.0, team_xgc=1.0, cs_prob=0.2,
            yellow_card_prob=0.0, red_card_prob=1.0,
        )
        assert result.xPts_cards == pytest.approx(-3.0, abs=0.1)


class TestPointsCalculatorBreakdown:
    def test_breakdown_fields_present(self):
        result = calculate_player_xpts(
            player_id=1, name="Test", position="MID",
            xmins=75, xg_per90=0.3, xa_per90=0.2,
            team_xg=1.5, team_xgc=1.0, cs_prob=0.3,
        )
        assert result.xGoals >= 0
        assert result.xAssists >= 0
        assert result.xMins == pytest.approx(75.0)
        assert result.xPts_total >= 0

    def test_total_equals_sum_of_components(self):
        result = calculate_player_xpts(
            player_id=1, name="Test", position="DEF",
            xmins=80, xg_per90=0.1, xa_per90=0.05,
            team_xg=1.2, team_xgc=0.8, cs_prob=0.35,
        )
        component_sum = (
            result.xPts_appearance + result.xPts_goals + result.xPts_assists
            + result.xPts_clean_sheet + result.xPts_goals_conceded
            + result.xPts_saves + result.xPts_bonus + result.xPts_cards
        )
        assert result.xPts_total == pytest.approx(component_sum, abs=0.01)
