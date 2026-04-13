"""Transfer optimiser: find the best 1-3 transfers given a current squad."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

# Points hit per additional transfer beyond free transfers
HIT_COST = 4


@dataclass
class TransferRecommendation:
    """A single transfer recommendation."""

    player_out: str
    player_out_id: int
    player_out_xpts: float
    player_out_price: float

    player_in: str
    player_in_id: int
    player_in_xpts: float
    player_in_price: float

    position: str
    xpts_gain: float
    hit_cost: float
    net_gain: float
    price_diff: float


@dataclass
class TransferPlan:
    """A multi-transfer plan (1, 2, or 3 transfers)."""

    transfers: list[TransferRecommendation]
    total_xpts_gain: float
    total_hit_cost: float
    net_gain: float
    free_transfers_used: int
    hits_taken: int


def optimise_transfers(
    current_squad: pd.DataFrame,
    all_players: pd.DataFrame,
    xpts_col: str = "total_xPts",
    n_transfers: int = 1,
    free_transfers: int = 1,
    budget_remaining: float = 0.0,
    max_per_club: int = 3,
) -> list[TransferPlan]:
    """Find the best transfer plans for 1, 2, or 3 transfers.

    For each number of transfers up to n_transfers, finds the plan that
    maximises net xPts gain after accounting for points hits.

    Args:
        current_squad: 15-player squad DataFrame with id, position, price, xpts_col.
        all_players: Full player pool DataFrame (same schema).
        xpts_col: Projected points column.
        n_transfers: Maximum transfers to consider (1, 2, or 3).
        free_transfers: Number of free transfers available.
        budget_remaining: Additional transfer budget in £m (ITB).
        max_per_club: Max players from same club (3 by default).

    Returns:
        List of TransferPlan objects sorted by net gain descending.
    """
    n_transfers = min(n_transfers, 3)
    plans: list[TransferPlan] = []

    for k in range(1, n_transfers + 1):
        plan = _find_best_k_transfers(
            current_squad,
            all_players,
            xpts_col,
            k,
            free_transfers,
            budget_remaining,
            max_per_club,
        )
        if plan is not None:
            plans.append(plan)

    plans.sort(key=lambda p: p.net_gain, reverse=True)
    return plans


def _find_best_k_transfers(
    squad: pd.DataFrame,
    pool: pd.DataFrame,
    xpts_col: str,
    k: int,
    free_transfers: int,
    budget_remaining: float,
    max_per_club: int,
) -> TransferPlan | None:
    """Find the best combination of exactly k transfers."""
    squad = squad.copy().reset_index(drop=True)
    pool = pool.copy().reset_index(drop=True)

    # Exclude current squad members from the candidate pool
    squad_ids = set(squad["id"].tolist())
    candidates = pool[~pool["id"].isin(squad_ids)].copy()

    hits = max(0, k - free_transfers)
    hit_cost = hits * HIT_COST

    # Group candidates by position for fast lookup
    by_pos: dict[str, pd.DataFrame] = {
        pos: candidates[candidates["position"] == pos].copy()
        for pos in ["GK", "DEF", "MID", "FWD"]
    }

    best_gain = -999.0
    best_transfers: list[TransferRecommendation] | None = None

    # Enumerate all combinations of k players to remove
    for out_combo in combinations(squad.itertuples(index=True), k):
        # Check we're not removing too many from one position
        out_positions = [r.position for r in out_combo]
        out_ids = {r.id for r in out_combo}

        # Build the squad after removals
        remaining = squad[~squad["id"].isin(out_ids)]

        # For each position being filled, find the best available player
        # Ensure positional slot constraints remain satisfied after the swap
        total_gain = 0.0
        transfer_list: list[TransferRecommendation] = []
        used_in_ids: set[int] = set()
        feasible = True

        # Budget available = sum of sold players' prices + ITB
        sell_value = sum(r.price for r in out_combo)
        available_budget = sell_value + budget_remaining

        # Sort out_combo by position for deterministic processing
        sorted_out = sorted(out_combo, key=lambda r: r.position)
        spend_so_far = 0.0

        for out_player in sorted_out:
            pos = out_player.position
            pos_candidates = by_pos.get(pos, pd.DataFrame())

            if pos_candidates.empty:
                feasible = False
                break

            # Budget for this slot
            slot_budget = available_budget - spend_so_far - sum(
                r.price for r in sorted_out
                if r != out_player
                # (already accounted in sell_value)
            )
            # Simplified budget: available_budget / k per slot (approximate)
            remaining_slots = len(sorted_out) - sorted_out.index(out_player)
            slot_budget = (available_budget - spend_so_far) - (
                # reserve budget for remaining slots: cheapest available
                sum(
                    by_pos.get(sorted_out[j].position, pd.DataFrame())
                    .loc[~by_pos.get(sorted_out[j].position, pd.DataFrame())["id"].isin(used_in_ids)]
                    ["price"].min()
                    if not by_pos.get(sorted_out[j].position, pd.DataFrame()).empty else 0
                    for j in range(sorted_out.index(out_player) + 1, len(sorted_out))
                )
            )

            # Club limit check: remaining squad after removals + players already chosen
            club_counts = remaining["team"].value_counts().to_dict()
            for uid in used_in_ids:
                t = pool.loc[pool["id"] == uid, "team"].iloc[0] if uid in pool["id"].values else ""
                club_counts[t] = club_counts.get(t, 0) + 1

            eligible = pos_candidates[
                (~pos_candidates["id"].isin(used_in_ids))
                & (pos_candidates["price"] <= slot_budget)
                & (pos_candidates.apply(
                    lambda r: club_counts.get(r["team"], 0) < max_per_club, axis=1
                ))
            ]

            if eligible.empty:
                feasible = False
                break

            best_in = eligible.nlargest(1, xpts_col).iloc[0]
            gain = float(best_in[xpts_col]) - float(out_player.__getattribute__(xpts_col) if hasattr(out_player, xpts_col) else 0)
            total_gain += gain
            spend_so_far += float(best_in["price"])
            used_in_ids.add(int(best_in["id"]))

            transfer_list.append(
                TransferRecommendation(
                    player_out=out_player.web_name if hasattr(out_player, "web_name") else out_player.name,
                    player_out_id=int(out_player.id),
                    player_out_xpts=float(getattr(out_player, xpts_col, 0)),
                    player_out_price=float(out_player.price),
                    player_in=best_in.get("web_name", best_in.get("name", "")),
                    player_in_id=int(best_in["id"]),
                    player_in_xpts=float(best_in[xpts_col]),
                    player_in_price=float(best_in["price"]),
                    position=pos,
                    xpts_gain=round(float(best_in[xpts_col]) - float(getattr(out_player, xpts_col, 0)), 2),
                    hit_cost=0.0,
                    net_gain=0.0,
                    price_diff=round(float(best_in["price"]) - float(out_player.price), 1),
                )
            )

        if not feasible:
            continue

        net = total_gain - hit_cost
        if net > best_gain:
            best_gain = net
            best_transfers = transfer_list

    if best_transfers is None:
        logger.warning(f"No feasible {k}-transfer plan found")
        return None

    # Assign hit cost proportionally to last transfer(s)
    for i, tr in enumerate(best_transfers):
        tr.hit_cost = HIT_COST if i >= free_transfers else 0.0
        tr.net_gain = round(tr.xpts_gain - tr.hit_cost, 2)

    total_xpts_gain = sum(t.xpts_gain for t in best_transfers)

    return TransferPlan(
        transfers=best_transfers,
        total_xpts_gain=round(total_xpts_gain, 2),
        total_hit_cost=float(hit_cost),
        net_gain=round(total_xpts_gain - hit_cost, 2),
        free_transfers_used=min(k, free_transfers),
        hits_taken=hits,
    )


def format_transfer_plan(plan: TransferPlan) -> str:
    """Format a transfer plan for CLI display."""
    lines = [
        f"\n{'=' * 55}",
        f"  {plan.free_transfers_used} free transfer(s)"
        + (f"  +{plan.hits_taken} hit(s) (-{plan.total_hit_cost:.0f} pts)" if plan.hits_taken else ""),
        f"  Net gain: {plan.net_gain:+.1f} xPts",
        f"{'=' * 55}",
    ]
    for tr in plan.transfers:
        marker = "●"
        lines.append(
            f"  {marker} OUT: {tr.player_out:<22} £{tr.player_out_price:.1f}m  "
            f"({tr.player_out_xpts:.1f} xPts)"
        )
        lines.append(
            f"     IN:  {tr.player_in:<22} £{tr.player_in_price:.1f}m  "
            f"({tr.player_in_xpts:.1f} xPts)  [{tr.xpts_gain:+.1f}]"
        )
    return "\n".join(lines)
