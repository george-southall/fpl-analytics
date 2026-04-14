"""Build per-GW transfer feature tables for price change modelling.

Fetches per-GW history for every player via the FPL element-summary endpoint
and constructs the feature set used by the XGBoost price change classifier.

Key features
------------
net_transfer_rate   (transfers_in - transfers_out) / selected  — the primary
                    signal in the FPL price-change algorithm
ownership_pct       selected / ~8_000_000 (total FPL players, approximate)
value               price in tenths of £m (60 = £6.0m)
price_change_lag1   price delta versus previous GW (momentum)
net_rate_lag1       net_transfer_rate last GW
form5               rolling mean of total_points over last 5 GWs
position_enc        GK=0, DEF=1, MID=2, FWD=3
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from fpl_analytics.ingestion.fpl_api import FPLClient
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

TOTAL_FPL_PLAYERS = 8_000_000  # approximate; used for ownership_pct normalisation
POSITION_ENC = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
_ETYPE_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _fetch_player_history(player_id: int, client: FPLClient) -> list[dict]:
    """Return list of per-GW history dicts for one player."""
    try:
        summary = client.get_player_summary(player_id)
        return summary.get("history", [])
    except Exception:
        return []


def fetch_all_histories(
    players_df: pd.DataFrame,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Fetch per-GW histories for all players and return a flat DataFrame.

    Args:
        players_df: FPL player DataFrame (must have 'id' and 'element_type' columns).
        max_workers: parallel HTTP threads.

    Returns:
        DataFrame with one row per (player_id, round), raw API fields plus
        'element_type' and 'position'.
    """
    client = FPLClient()
    player_ids = players_df["id"].tolist()
    etype_map = players_df.set_index("id")["element_type"].to_dict()

    logger.info(f"Fetching GW histories for {len(player_ids)} players ({max_workers} threads)…")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_player_history, pid, client): pid for pid in player_ids}
        for future in as_completed(futures):
            pid = futures[future]
            history = future.result()
            for row in history:
                row["player_id"] = pid
                row["element_type"] = etype_map.get(pid, 0)
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["position"] = df["element_type"].map(_ETYPE_MAP).fillna("MID")
    df["round"] = df["round"].astype(int)
    df["value"] = df["value"].astype(float)
    df["selected"] = df["selected"].astype(float).clip(lower=1)
    df["transfers_in"] = df["transfers_in"].astype(float)
    df["transfers_out"] = df["transfers_out"].astype(float)
    df["total_points"] = df["total_points"].astype(float)
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True)

    logger.info(f"Fetched {len(df)} player-GW rows across {df['round'].nunique()} GWs")
    return df


def build_features(history_df: pd.DataFrame, label: bool = True) -> pd.DataFrame:
    """Compute model features (and optionally the target label) from raw history.

    Args:
        history_df: Output of fetch_all_histories().
        label: If True, compute 'target' column (+1 rise / 0 hold / -1 fall).
                Set False when predicting on the latest (incomplete) GW.

    Returns:
        Feature DataFrame.  Rows with NaN lags (first GW per player) are dropped
        when label=True.
    """
    df = history_df.copy()

    # Per-player lagged features
    df = df.sort_values(["player_id", "round"])
    grp = df.groupby("player_id")

    df["net_transfers"] = df["transfers_in"] - df["transfers_out"]
    df["net_transfer_rate"] = df["net_transfers"] / df["selected"]
    df["ownership_pct"] = df["selected"] / TOTAL_FPL_PLAYERS * 100

    df["price_change_lag1"] = grp["value"].diff()
    df["net_rate_lag1"] = grp["net_transfer_rate"].shift(1)
    df["value_lag1"] = grp["value"].shift(1)

    # Rolling form (last 5 GWs, min 1)
    df["form5"] = (
        grp["total_points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df["position_enc"] = df["position"].map(POSITION_ENC).fillna(2).astype(int)

    if label:
        # Target: next-GW price change sign
        df["value_next"] = grp["value"].shift(-1)
        df["price_change_next"] = df["value_next"] - df["value"]
        df["target"] = np.sign(df["price_change_next"]).fillna(0).astype(int)
        # Drop rows missing lags or target
        df = df.dropna(subset=FEATURE_COLS + ["target"])

    return df


FEATURE_COLS = [
    "net_transfer_rate",
    "net_rate_lag1",
    "ownership_pct",
    "value",
    "price_change_lag1",
    "form5",
    "position_enc",
    "total_points",
]


def current_gw_features(players_df: pd.DataFrame, max_workers: int = 10) -> pd.DataFrame:
    """Return a feature DataFrame for the *current* GW (for live prediction).

    Fetches histories, builds features for the most recent completed GW per
    player, and returns one row per player (merged with 'web_name', 'team_name',
    'price', 'selected_by_percent').
    """
    history_df = fetch_all_histories(players_df, max_workers=max_workers)
    if history_df.empty:
        return pd.DataFrame()

    feat = build_features(history_df, label=False)

    # Keep only the latest GW row per player
    latest = feat.loc[feat.groupby("player_id")["round"].idxmax()].copy()

    # Merge display columns from players_df
    display = players_df[["id", "web_name", "team_name", "element_type", "now_cost",
                           "selected_by_percent", "transfers_in_event",
                           "transfers_out_event"]].copy()
    display = display.rename(columns={"id": "player_id", "now_cost": "price_tenths"})
    display["price"] = display["price_tenths"] / 10.0
    display["selected_by_percent"] = display["selected_by_percent"].astype(float)

    merged = latest.merge(display, on="player_id", how="left")
    return merged
