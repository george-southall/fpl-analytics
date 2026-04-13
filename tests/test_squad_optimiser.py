"""Tests for the squad optimiser, transfer optimiser, and captain picker."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_analytics.optimiser.captain_picker import pick_captain
from fpl_analytics.optimiser.squad_optimiser import optimise_squad
from fpl_analytics.optimiser.transfer_optimiser import (
    HIT_COST,
    optimise_transfers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_player_pool(seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic 80-player pool (2 GK, 5 DEF, 5 MID, 3 FWD per club × 5 clubs)."""
    import numpy as np
    rng = np.random.default_rng(seed)

    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    clubs = ["Arsenal", "Liverpool", "Chelsea", "Newcastle", "Tottenham"]

    rows = []
    pid = 1
    for club in clubs:
        for pos in positions:
            base_price = {"GK": 5.0, "DEF": 5.5, "MID": 7.0, "FWD": 8.0}[pos]
            price = round(base_price + rng.uniform(-1, 3), 1)
            xpts = round(rng.uniform(2, 12), 2)
            rows.append({
                "id": pid,
                "web_name": f"{pos}_{club[:3]}_{pid}",
                "team": club,
                "position": pos,
                "price": price,
                "total_xPts": xpts,
                "selected_by_percent": round(rng.uniform(1, 50), 1),
            })
            pid += 1

    return pd.DataFrame(rows)


@pytest.fixture
def player_pool():
    return _make_player_pool(seed=42)


@pytest.fixture
def valid_squad(player_pool):
    """A valid 15-man squad from the pool."""
    sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
    return pd.concat([sol.starters, sol.bench]).reset_index(drop=True)


# ── Squad Optimiser Tests ─────────────────────────────────────────────────────

class TestSquadOptimiser:
    def test_returns_15_players(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        total = len(sol.starters) + len(sol.bench)
        assert total == 15

    def test_exactly_2_gks(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        squad = pd.concat([sol.starters, sol.bench])
        assert (squad["position"] == "GK").sum() == 2

    def test_exactly_5_defenders(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        squad = pd.concat([sol.starters, sol.bench])
        assert (squad["position"] == "DEF").sum() == 5

    def test_exactly_5_midfielders(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        squad = pd.concat([sol.starters, sol.bench])
        assert (squad["position"] == "MID").sum() == 5

    def test_exactly_3_forwards(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        squad = pd.concat([sol.starters, sol.bench])
        assert (squad["position"] == "FWD").sum() == 3

    def test_respects_budget(self, player_pool):
        budget = 95.0
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=budget)
        assert sol.total_cost <= budget + 0.01  # small float tolerance

    def test_max_3_per_club(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        squad = pd.concat([sol.starters, sol.bench])
        club_counts = squad["team"].value_counts()
        assert club_counts.max() <= 3

    def test_starting_xi_has_11_players(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        assert len(sol.starters) == 11

    def test_bench_has_4_players(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        assert len(sol.bench) == 4

    def test_exactly_1_gk_in_starting_xi(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        assert (sol.starters["position"] == "GK").sum() == 1

    def test_valid_formation(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        d, m, f = sol.formation
        assert d + m + f == 10  # 10 outfield starters
        assert 3 <= d <= 5
        assert 2 <= m <= 5
        assert 1 <= f <= 3

    def test_no_duplicate_players(self, player_pool):
        sol = optimise_squad(player_pool, xpts_col="total_xPts", budget=100.0)
        squad = pd.concat([sol.starters, sol.bench])
        assert squad["id"].nunique() == 15

    def test_tight_budget_raises_or_finds_solution(self, player_pool):
        """With an extremely tight budget the LP may be infeasible."""
        with pytest.raises((RuntimeError, Exception)):
            optimise_squad(player_pool, xpts_col="total_xPts", budget=30.0)


# ── Transfer Optimiser Tests ──────────────────────────────────────────────────

class TestTransferOptimiser:
    def test_1_transfer_returns_plan(self, player_pool, valid_squad):
        plans = optimise_transfers(
            valid_squad, player_pool, xpts_col="total_xPts",
            n_transfers=1, free_transfers=1,
        )
        assert len(plans) >= 1

    def test_1_free_transfer_no_hit(self, player_pool, valid_squad):
        plans = optimise_transfers(
            valid_squad, player_pool, xpts_col="total_xPts",
            n_transfers=1, free_transfers=1,
        )
        best = plans[0]
        assert best.hits_taken == 0
        assert best.total_hit_cost == 0.0

    def test_2_transfers_with_1_free_takes_hit(self, player_pool, valid_squad):
        plans = optimise_transfers(
            valid_squad, player_pool, xpts_col="total_xPts",
            n_transfers=2, free_transfers=1,
        )
        two_transfer_plans = [p for p in plans if len(p.transfers) == 2]
        if two_transfer_plans:
            plan = two_transfer_plans[0]
            assert plan.hits_taken == 1
            assert plan.total_hit_cost == HIT_COST

    def test_transfers_respect_positional_slots(self, player_pool, valid_squad):
        plans = optimise_transfers(
            valid_squad, player_pool, xpts_col="total_xPts",
            n_transfers=1, free_transfers=1,
        )
        for plan in plans:
            for tr in plan.transfers:
                assert tr.position in ("GK", "DEF", "MID", "FWD")

    def test_transfer_in_not_already_in_squad(self, player_pool, valid_squad):
        squad_ids = set(valid_squad["id"].tolist())
        plans = optimise_transfers(
            valid_squad, player_pool, xpts_col="total_xPts",
            n_transfers=1, free_transfers=1,
        )
        for plan in plans:
            for tr in plan.transfers:
                assert tr.player_in_id not in squad_ids


# ── Captain Picker Tests ──────────────────────────────────────────────────────

class TestCaptainPicker:
    def test_captain_is_highest_xpts(self, valid_squad):
        rec = pick_captain(valid_squad, xpts_col="total_xPts")
        max_xpts = valid_squad["total_xPts"].max()
        assert rec.captain_xpts == pytest.approx(max_xpts, abs=0.01)

    def test_vice_is_second_highest(self, valid_squad):
        rec = pick_captain(valid_squad, xpts_col="total_xPts")
        sorted_xpts = valid_squad["total_xPts"].sort_values(ascending=False).values
        assert rec.vice_captain_xpts == pytest.approx(sorted_xpts[1], abs=0.01)

    def test_captain_and_vice_are_different(self, valid_squad):
        rec = pick_captain(valid_squad, xpts_col="total_xPts")
        assert rec.captain_id != rec.vice_captain_id

    def test_differential_captain_below_ownership_threshold(self, valid_squad):
        rec = pick_captain(valid_squad, xpts_col="total_xPts", differential_threshold=10.0)
        if rec.differential_captain is not None:
            assert rec.differential_captain_ownership < 10.0

    def test_candidates_table_present(self, valid_squad):
        rec = pick_captain(valid_squad, xpts_col="total_xPts")
        assert rec.all_candidates is not None
        assert len(rec.all_candidates) > 0

    def test_ev_bonus_equals_xpts(self, valid_squad):
        rec = pick_captain(valid_squad, xpts_col="total_xPts")
        assert rec.captain_ev_bonus == pytest.approx(rec.captain_xpts, abs=0.01)
