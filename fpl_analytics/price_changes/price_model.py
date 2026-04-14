"""XGBoost price change classifier.

Predicts whether a player's FPL price will rise (+1), hold (0), or fall (-1)
in the next gameweek based on per-GW transfer and performance features.

Training uses walk-forward cross-validation to respect temporal ordering.
The model uses XGBoost's native CUDA device support when available.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from fpl_analytics.config import settings
from fpl_analytics.price_changes.net_transfers import FEATURE_COLS
from fpl_analytics.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = settings.data_dir / "models" / "price_change_model.pkl"
LABEL_MAP = {-1: "fall", 0: "hold", 1: "rise"}
LABEL_MAP_INV = {"fall": -1, "hold": 0, "rise": 1}


def _make_xgb(use_gpu: bool = True) -> XGBClassifier:
    """Instantiate XGBClassifier, using CUDA if available and requested."""
    device = "cuda" if use_gpu else "cpu"
    # XGBoost ≥ 2.0 uses device= parameter
    try:
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            device=device,
            random_state=42,
            n_jobs=-1,
        )
    except TypeError:
        # Older XGBoost; fall back without device param
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    return clf


class PriceChangeModel:
    """XGBoost classifier for FPL price change prediction.

    Labels: +1 = price rise, 0 = hold, -1 = fall.
    """

    def __init__(self) -> None:
        self.clf: XGBClassifier | None = None
        self._le = LabelEncoder()
        self._feature_cols = FEATURE_COLS
        self._trained = False
        self.precision_: float | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, features_df: pd.DataFrame, use_gpu: bool = True) -> "PriceChangeModel":
        """Train on labelled feature data with walk-forward cross-validation.

        Args:
            features_df: Output of build_features(label=True).
            use_gpu: Attempt CUDA training.

        Returns:
            self (fitted).
        """
        df = features_df.dropna(subset=self._feature_cols + ["target", "round"])
        rounds = sorted(df["round"].unique())
        n = len(rounds)

        if n < 5:
            raise ValueError(f"Need ≥ 5 GWs of data to train; got {n}")

        # Walk-forward CV: train on first 75% of rounds, validate on remainder
        split_idx = max(3, int(n * 0.75))
        train_rounds = rounds[:split_idx]
        val_rounds = rounds[split_idx:]

        train = df[df["round"].isin(train_rounds)]
        val = df[df["round"].isin(val_rounds)]

        X_train = train[self._feature_cols].values
        y_train = self._le.fit_transform(train["target"].values)
        X_val = val[self._feature_cols].values
        y_val = self._le.transform(val["target"].values)

        logger.info(
            f"Training price model: {len(X_train)} train rows ({len(train_rounds)} GWs), "
            f"{len(X_val)} val rows ({len(val_rounds)} GWs)"
        )

        self.clf = _make_xgb(use_gpu=use_gpu)
        try:
            self.clf.fit(X_train, y_train)
        except Exception as exc:
            if use_gpu and "cuda" in str(exc).lower():
                logger.warning(f"GPU training failed ({exc}), retrying on CPU")
                self.clf = _make_xgb(use_gpu=False)
                self.clf.fit(X_train, y_train)
            else:
                raise

        # Validation precision (macro average)
        y_pred_val = self.clf.predict(X_val)
        self.precision_ = precision_score(y_val, y_pred_val, average="macro", zero_division=0)
        logger.info(f"Validation macro precision: {self.precision_:.3f}")
        logger.info("\n" + classification_report(
            y_val, y_pred_val,
            target_names=[str(c) for c in self._le.classes_],
            zero_division=0,
        ))

        # Refit on full dataset
        X_all = df[self._feature_cols].values
        y_all = self._le.transform(df["target"].values)
        self.clf.fit(X_all, y_all)
        self._trained = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba_df(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of class probabilities for each player row.

        Columns: prob_fall, prob_hold, prob_rise, prediction (-1/0/1).
        """
        if not self._trained or self.clf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = features_df[self._feature_cols].fillna(0).values
        probs = self.clf.predict_proba(X)  # shape (N, n_classes)
        classes = self._le.classes_  # e.g. [-1, 0, 1]

        out = features_df.copy()
        class_labels = {c: f"prob_{LABEL_MAP[c]}" for c in classes}
        for i, cls in enumerate(classes):
            out[class_labels[cls]] = probs[:, i]

        out["prediction"] = self._le.inverse_transform(self.clf.predict(X))

        # Ensure all three columns exist even if class not seen in training
        for col in ["prob_fall", "prob_hold", "prob_rise"]:
            if col not in out.columns:
                out[col] = 0.0

        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Price change model saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "PriceChangeModel":
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def is_trained(self) -> bool:
        return self._trained
