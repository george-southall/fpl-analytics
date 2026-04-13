"""
Refresh all data sources.

Usage:
    python -m fpl_analytics.ingestion.refresh
    python -m fpl_analytics.ingestion.refresh --skip-understat
    python -m fpl_analytics.ingestion.refresh --fpl-only
"""

from __future__ import annotations

import argparse
import sys
import time

from fpl_analytics.db import init_db
from fpl_analytics.utils.logger import get_logger

logger = get_logger("fpl_analytics.refresh")


def refresh_fpl() -> None:
    """Refresh FPL API data (players, teams, fixtures)."""
    from fpl_analytics.ingestion.fpl_api import FPLClient

    client = FPLClient()
    client.persist_players()
    client.persist_teams()
    client.persist_fixtures()


def refresh_results() -> None:
    """Refresh historical match results."""
    from fpl_analytics.ingestion.results_fetcher import fetch_all_seasons, persist_results

    df = fetch_all_seasons()
    persist_results(df)


def refresh_understat() -> None:
    """Refresh Understat xG/xA data."""
    from fpl_analytics.ingestion.understat_fetcher import fetch_current_season, persist_understat

    df = fetch_current_season()
    persist_understat(df)


def refresh_all(skip_understat: bool = False, fpl_only: bool = False) -> None:
    """Refresh all data sources."""
    init_db()
    start = time.time()

    logger.info("=" * 60)
    logger.info("FPL Analytics — Data Refresh")
    logger.info("=" * 60)

    # FPL API
    logger.info("\n[1/3] FPL API data...")
    try:
        refresh_fpl()
    except Exception as e:
        logger.error(f"FPL API refresh failed: {e}")
        raise

    if fpl_only:
        logger.info(f"\nFPL-only refresh complete in {time.time() - start:.1f}s")
        return

    # Historical results
    logger.info("\n[2/3] Historical match results...")
    try:
        refresh_results()
    except Exception as e:
        logger.error(f"Results refresh failed: {e}")
        raise

    # Understat
    if not skip_understat:
        logger.info("\n[3/3] Understat xG/xA data...")
        try:
            refresh_understat()
        except Exception as e:
            logger.error(f"Understat refresh failed: {e}")
            logger.warning("Continuing without Understat data")
    else:
        logger.info("\n[3/3] Skipping Understat (--skip-understat)")

    # Validation
    logger.info("\n[✓] Running data validation...")
    try:
        _run_validation()
    except Exception as e:
        logger.warning(f"Validation encountered issues: {e}")

    elapsed = time.time() - start
    logger.info(f"\nData refresh complete in {elapsed:.1f}s")


def _run_validation() -> None:
    """Run validation checks on freshly loaded data."""
    from fpl_analytics.ingestion.data_validator import DataValidator
    from fpl_analytics.ingestion.fpl_api import FPLClient

    client = FPLClient()
    validator = DataValidator()

    players_df = client.get_players_df()
    fixtures_df = client.get_fixtures_df()

    validator.validate_players(players_df)
    validator.validate_fixtures(fixtures_df)
    summary = validator.summary()

    if not summary["all_passed"]:
        logger.warning(f"Validation: {summary['failed']} check(s) failed — review above")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh FPL Analytics data sources")
    parser.add_argument("--skip-understat", action="store_true", help="Skip Understat fetch")
    parser.add_argument("--fpl-only", action="store_true", help="Only refresh FPL API data")
    args = parser.parse_args()

    try:
        refresh_all(skip_understat=args.skip_understat, fpl_only=args.fpl_only)
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
