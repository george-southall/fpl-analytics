"""Generate price change alerts combining the XGBoost model with heuristics.

Provides a unified interface for the dashboard:
  - load_or_train_model(): returns a fitted PriceChangeModel, training on-the-fly
    the first time (or when the saved model is stale).
  - generate_alerts(): returns a player-level alert DataFrame with model
    probabilities and a heuristic alert flag.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from fpl_analytics.price_changes.net_transfers import (
    FEATURE_COLS,
    build_features,
    current_gw_features,
    fetch_all_histories,
)
from fpl_analytics.price_changes.price_model import MODEL_PATH, PriceChangeModel
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

# Heuristic threshold: net_transfer_rate > RISE_THRESH → likely rise candidate
RISE_THRESH = 0.005   # 0.5% of owners in a single GW
FALL_THRESH = -0.005


def load_or_train_model(
    players_df: pd.DataFrame,
    force_retrain: bool = False,
    max_age_hours: float = 24.0,
    use_gpu: bool = True,
) -> PriceChangeModel:
    """Load saved model if fresh; otherwise train and save.

    Args:
        players_df: FPL player DataFrame (used to fetch training histories).
        force_retrain: Skip cache check and always retrain.
        max_age_hours: Retrain if saved model is older than this.
        use_gpu: Use CUDA for XGBoost training.

    Returns:
        Fitted PriceChangeModel.
    """
    if not force_retrain and MODEL_PATH.exists():
        age_hours = (time.time() - MODEL_PATH.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            logger.info(f"Loading cached price model (age {age_hours:.1f}h)")
            try:
                model = PriceChangeModel.load()
                if model.is_trained:
                    return model
            except Exception as exc:
                logger.warning(f"Failed to load cached model ({exc}), retraining")

    logger.info("Training price change model on current-season GW histories…")
    history_df = fetch_all_histories(players_df)
    if history_df.empty:
        logger.warning("No history data available; returning untrained model")
        return PriceChangeModel()

    features = build_features(history_df, label=True)
    if len(features) < 100:
        logger.warning(f"Only {len(features)} labelled rows; model quality may be low")

    model = PriceChangeModel()
    model.fit(features, use_gpu=use_gpu)
    model.save()
    return model


def generate_alerts(
    players_df: pd.DataFrame,
    model: PriceChangeModel | None = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Produce a player-level price alert DataFrame.

    Columns returned
    ----------------
    player_id, web_name, team_name, position, price,
    selected_by_percent, net_transfers_event,
    net_transfer_rate,          # current GW rate
    heuristic_alert,            # 'rise' | 'fall' | None (threshold-based)
    prob_rise, prob_hold, prob_fall,  # XGBoost probabilities (0 if model untrained)
    prediction,                 # model's top class: -1 / 0 / 1
    confidence,                 # max probability across classes
    alert                       # combined alert: 'rise' | 'fall' | 'hold'
    """
    feat_df = current_gw_features(players_df, max_workers=max_workers)
    if feat_df.empty:
        return pd.DataFrame()

    # Heuristic alert
    feat_df["net_transfers_event"] = (
        feat_df["transfers_in_event"] - feat_df["transfers_out_event"]
    )
    feat_df["net_transfer_rate_event"] = (
        feat_df["net_transfers_event"] / feat_df["selected"].clip(lower=1)
    )
    feat_df["heuristic_alert"] = None
    feat_df.loc[feat_df["net_transfer_rate_event"] >= RISE_THRESH, "heuristic_alert"] = "rise"
    feat_df.loc[feat_df["net_transfer_rate_event"] <= FALL_THRESH, "heuristic_alert"] = "fall"

    # XGBoost prediction
    if model is not None and model.is_trained:
        feat_df = feat_df.fillna({c: 0 for c in FEATURE_COLS})
        feat_df = model.predict_proba_df(feat_df)
        feat_df["confidence"] = feat_df[["prob_fall", "prob_hold", "prob_rise"]].max(axis=1)
        # Combined alert: prefer model when confident (>0.5), else heuristic
        def _combined(row):
            if row.get("confidence", 0) >= 0.5:
                p = row.get("prediction", 0)
                return {1: "rise", -1: "fall", 0: "hold"}.get(p, "hold")
            return row.get("heuristic_alert") or "hold"

        feat_df["alert"] = feat_df.apply(_combined, axis=1)
    else:
        feat_df["prob_rise"] = 0.0
        feat_df["prob_hold"] = 1.0
        feat_df["prob_fall"] = 0.0
        feat_df["prediction"] = 0
        feat_df["confidence"] = 0.0
        feat_df["alert"] = feat_df["heuristic_alert"].fillna("hold")

    # Tidy output
    keep = [
        "player_id", "web_name", "team_name", "position", "price",
        "selected_by_percent", "net_transfers_event", "net_transfer_rate_event",
        "heuristic_alert", "prob_rise", "prob_hold", "prob_fall",
        "prediction", "confidence", "alert",
    ]
    keep = [c for c in keep if c in feat_df.columns]
    return feat_df[keep].reset_index(drop=True)
