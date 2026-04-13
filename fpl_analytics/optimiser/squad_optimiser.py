"""Squad optimiser using PuLP linear programming.

Selects the optimal 15-man FPL squad (11 starters + 4 bench) that maximises
projected points subject to budget, positional, and club constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from fpl_analytics.config import settings
from fpl_analytics.utils.fpl_constants import (
    SQUAD_COMPOSITION,
    VALID_FORMATIONS,
    POSITION_ID,
)
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SquadSolution:
    """Result of squad optimisation."""

    starters: pd.DataFrame
    bench: pd.DataFrame
    total_cost: float
    total_xpts: float
    formation: tuple[int, int, int]
    status: str


def _position_counts(df: pd.DataFrame) -> dict[str, int]:
    return df["position"].value_counts().to_dict()


def optimise_squad(
    players_df: pd.DataFrame,
    xpts_col: str = "total_xPts",
    budget: float | None = None,
    max_per_club: int | None = None,
) -> SquadSolution:
    """Select the optimal 15-man FPL squad.

    Args:
        players_df: Must contain columns: id, position, team, price, and xpts_col.
        xpts_col: Column name for projected points to maximise.
        budget: Total budget in £m (default: settings.squad_budget).
        max_per_club: Max players from same club (default: settings.max_players_per_club).

    Returns:
        SquadSolution with starters, bench, cost, and xpts.
    """
    budget = budget or settings.squad_budget
    max_per_club = max_per_club or settings.max_players_per_club

    df = players_df.copy().reset_index(drop=True)
    n = len(df)
    ids = list(range(n))

    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

    # Decision variables: x[i] = 1 if player i is in the 15-man squad
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids]

    # Objective: maximise total projected points
    prob += pulp.lpSum(float(df.loc[i, xpts_col]) * x[i] for i in ids)

    # ── Constraints ────────────────────────────────────────────────────────────

    # Total squad size = 15
    prob += pulp.lpSum(x) == 15

    # Positional constraints (exact squad composition)
    for pos, count in SQUAD_COMPOSITION.items():
        pos_ids = [i for i in ids if df.loc[i, "position"] == pos]
        prob += pulp.lpSum(x[i] for i in pos_ids) == count

    # Budget constraint
    prob += pulp.lpSum(float(df.loc[i, "price"]) * x[i] for i in ids) <= budget

    # Max players per club
    for club in df["team"].unique():
        club_ids = [i for i in ids if df.loc[i, "team"] == club]
        prob += pulp.lpSum(x[i] for i in club_ids) <= max_per_club

    # ── Solve ──────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    if prob.status != 1:
        raise RuntimeError(f"Squad optimisation failed: {status}")

    selected_idx = [i for i in ids if pulp.value(x[i]) > 0.5]
    squad_df = df.loc[selected_idx].copy()

    logger.info(
        f"Squad optimised [{status}]: "
        f"£{squad_df['price'].sum():.1f}m, "
        f"{squad_df[xpts_col].sum():.1f} xPts"
    )

    # ── Two-pass: optimise starting XI from the 15 ────────────────────────────
    starters, bench = _optimise_starting_xi(squad_df, xpts_col)
    formation = _detect_formation(starters)

    return SquadSolution(
        starters=starters.reset_index(drop=True),
        bench=bench.reset_index(drop=True),
        total_cost=round(squad_df["price"].sum(), 1),
        total_xpts=round(squad_df[xpts_col].sum(), 2),
        formation=formation,
        status=status,
    )


def _optimise_starting_xi(
    squad_df: pd.DataFrame,
    xpts_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the best 11 starters from the 15-man squad."""
    df = squad_df.copy().reset_index(drop=True)
    n = len(df)
    ids = list(range(n))

    prob = pulp.LpProblem("FPL_XI", pulp.LpMaximize)
    s = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in ids]

    prob += pulp.lpSum(float(df.loc[i, xpts_col]) * s[i] for i in ids)

    # Exactly 11 starters
    prob += pulp.lpSum(s) == 11

    # Exactly 1 GK in starting XI
    gk_ids = [i for i in ids if df.loc[i, "position"] == "GK"]
    prob += pulp.lpSum(s[i] for i in gk_ids) == 1

    # Valid formation: outfield constraints
    def_ids = [i for i in ids if df.loc[i, "position"] == "DEF"]
    mid_ids = [i for i in ids if df.loc[i, "position"] == "MID"]
    fwd_ids = [i for i in ids if df.loc[i, "position"] == "FWD"]

    # Formation bounds
    prob += pulp.lpSum(s[i] for i in def_ids) >= 3
    prob += pulp.lpSum(s[i] for i in def_ids) <= 5
    prob += pulp.lpSum(s[i] for i in mid_ids) >= 2
    prob += pulp.lpSum(s[i] for i in mid_ids) <= 5
    prob += pulp.lpSum(s[i] for i in fwd_ids) >= 1
    prob += pulp.lpSum(s[i] for i in fwd_ids) <= 3

    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if prob.status != 1:
        # Fall back: take top 11 by xPts respecting only the GK constraint
        starters = df.sort_values(xpts_col, ascending=False)
        gk_starter = starters[starters["position"] == "GK"].iloc[:1]
        outfield = starters[starters["position"] != "GK"].iloc[:10]
        starters = pd.concat([gk_starter, outfield])
        bench = df[~df.index.isin(starters.index)]
        return starters, bench

    starter_idx = [i for i in ids if pulp.value(s[i]) > 0.5]
    bench_idx = [i for i in ids if i not in starter_idx]

    starters = df.loc[starter_idx]
    bench = df.loc[bench_idx]

    # Order bench: GK last, rest by descending xPts
    bench_gk = bench[bench["position"] == "GK"]
    bench_out = bench[bench["position"] != "GK"].sort_values(xpts_col, ascending=False)
    bench = pd.concat([bench_out, bench_gk])

    return starters, bench


def _detect_formation(starters: pd.DataFrame) -> tuple[int, int, int]:
    """Detect the formation (DEF, MID, FWD) from the starting XI."""
    counts = starters["position"].value_counts()
    return (
        int(counts.get("DEF", 0)),
        int(counts.get("MID", 0)),
        int(counts.get("FWD", 0)),
    )
