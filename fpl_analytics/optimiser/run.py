"""CLI entry point for the squad/transfer optimiser.

Usage:
    python -m fpl_analytics.optimiser.run --transfers 1
    python -m fpl_analytics.optimiser.run --transfers 2 --free-transfers 2
    python -m fpl_analytics.optimiser.run --squad-only --budget 99.5
"""

from __future__ import annotations

import argparse
import sys

from fpl_analytics.utils.logger import get_logger

logger = get_logger("fpl_analytics.optimiser")


def main() -> None:
    parser = argparse.ArgumentParser(description="FPL Squad & Transfer Optimiser")
    parser.add_argument("--transfers", type=int, default=1, help="Max transfers to consider")
    parser.add_argument("--free-transfers", type=int, default=1, help="Free transfers available")
    parser.add_argument("--budget", type=float, default=None, help="Squad budget in £m")
    parser.add_argument("--squad-only", action="store_true", help="Only optimise squad, skip transfers")
    parser.add_argument("--gws", type=int, default=1, help="GW horizon for projections (1, 3, or 6)")
    args = parser.parse_args()

    try:
        _run(args)
    except Exception as e:
        logger.error(f"Optimiser failed: {e}")
        raise
        sys.exit(1)


def _run(args) -> None:
    from fpl_analytics.config import settings
    from fpl_analytics.db import init_db
    from fpl_analytics.ingestion.fpl_api import FPLClient
    from fpl_analytics.ingestion.results_fetcher import apply_time_decay, fetch_all_seasons
    from fpl_analytics.models.dixon_coles import DixonColesModel
    from fpl_analytics.models.team_strengths import TeamStrengths
    from fpl_analytics.optimiser.captain_picker import format_captain_recommendation, pick_captain
    from fpl_analytics.optimiser.squad_optimiser import optimise_squad
    from fpl_analytics.optimiser.transfer_optimiser import format_transfer_plan, optimise_transfers
    from fpl_analytics.projections.projection_engine import run_projections

    init_db()
    client = FPLClient()

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading FPL data...")
    players_df = client.get_players_df()
    fixtures_df = client.get_fixtures_df()

    logger.info("Fitting Dixon-Coles model...")
    results_df = fetch_all_seasons()
    results_df = apply_time_decay(results_df)
    ts = TeamStrengths()
    ts.fit(results_df)
    model = ts.model

    # ── Run projections ──────────────────────────────────────────────────────
    logger.info(f"Projecting over {args.gws} GW(s)...")
    projections = run_projections(players_df, fixtures_df, model, horizon=args.gws)

    xpts_col = "total_xPts"

    # ── Squad optimisation ───────────────────────────────────────────────────
    logger.info("Optimising squad...")
    solution = optimise_squad(projections, xpts_col=xpts_col, budget=args.budget)

    squad = solution.starters._append(solution.bench)
    formation = solution.formation
    print(f"\n{'=' * 55}")
    print(f"  OPTIMAL SQUAD  |  £{solution.total_cost:.1f}m  |  {solution.total_xpts:.1f} xPts")
    print(f"  Formation: {formation[0]}-{formation[1]}-{formation[2]}")
    print(f"{'=' * 55}")

    for _, p in solution.starters.iterrows():
        name = p.get("web_name") or p.get("name", "")
        print(f"  {p['position']:<4} {name:<25} £{p['price']:.1f}m  {p[xpts_col]:.1f} xPts")
    print("  --- Bench ---")
    for _, p in solution.bench.iterrows():
        name = p.get("web_name") or p.get("name", "")
        print(f"  {p['position']:<4} {name:<25} £{p['price']:.1f}m  {p[xpts_col]:.1f} xPts")

    # ── Captain recommendation ───────────────────────────────────────────────
    cap_rec = pick_captain(squad, xpts_col=xpts_col)
    print(format_captain_recommendation(cap_rec))

    if args.squad_only:
        return

    # ── Transfer optimisation ────────────────────────────────────────────────
    logger.info("Fetching your current squad...")
    try:
        my_team = client.get_my_team()
        picks = my_team.get("picks", [])
        my_ids = {p["element"] for p in picks}
        my_squad = projections[projections["id"].isin(my_ids)].copy()
    except Exception:
        logger.warning("Could not fetch your team — using optimised squad for transfer demo")
        my_squad = squad.copy()

    plans = optimise_transfers(
        my_squad,
        projections,
        xpts_col=xpts_col,
        n_transfers=args.transfers,
        free_transfers=args.free_transfers,
    )

    print(f"\n  TRANSFER RECOMMENDATIONS (up to {args.transfers} transfer(s))\n")
    for plan in plans:
        print(format_transfer_plan(plan))


if __name__ == "__main__":
    main()
