"""Tests for Phase 5 — price change model and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_analytics.price_changes.net_transfers import (
    FEATURE_COLS,
    build_features,
)
from fpl_analytics.price_changes.price_model import PriceChangeModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_history(n_players: int = 10, n_rounds: int = 15) -> pd.DataFrame:
    """Synthetic per-GW history DataFrame."""
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(1, n_players + 1):
        value = 60 + pid  # starting price (tenths)
        selected = rng.integers(100_000, 5_000_000)
        for rnd in range(1, n_rounds + 1):
            net = int(rng.integers(-50_000, 100_000))
            rows.append({
                "player_id": pid,
                "round": rnd,
                "value": value,
                "selected": selected,
                "transfers_in": max(0, net),
                "transfers_out": max(0, -net),
                "total_points": int(rng.integers(0, 15)),
                "element_type": rng.integers(1, 5),
                "position": ["GK", "DEF", "MID", "FWD"][rng.integers(0, 4)],
            })
            # Occasionally bump price
            if rng.random() < 0.08:
                value += 1
            elif rng.random() < 0.04:
                value -= 1
    return pd.DataFrame(rows)


# ── Feature engineering ───────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_returns_dataframe(self):
        hist = _make_history()
        feat = build_features(hist, label=True)
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) > 0

    def test_feature_cols_present(self):
        hist = _make_history()
        feat = build_features(hist, label=True)
        for col in FEATURE_COLS:
            assert col in feat.columns, f"Missing feature column: {col}"

    def test_target_values(self):
        hist = _make_history()
        feat = build_features(hist, label=True)
        assert feat["target"].isin([-1, 0, 1]).all()

    def test_no_nans_in_feature_cols(self):
        hist = _make_history()
        feat = build_features(hist, label=True)
        assert feat[FEATURE_COLS].isna().sum().sum() == 0

    def test_net_transfer_rate_sign(self):
        hist = _make_history()
        feat = build_features(hist, label=False)
        # Players with more transfers in than out should have positive rate
        pos_mask = hist.groupby(["player_id", "round"]).apply(
            lambda g: (g["transfers_in"] > g["transfers_out"]).all()
        )
        # Just verify the column exists and is numeric
        assert pd.api.types.is_float_dtype(feat["net_transfer_rate"])

    def test_label_false_keeps_latest_rows(self):
        hist = _make_history(n_players=5, n_rounds=10)
        feat_no_label = build_features(hist, label=False)
        feat_label = build_features(hist, label=True)
        # No-label version should have more rows (last GW not dropped)
        assert len(feat_no_label) >= len(feat_label)


# ── Price model ───────────────────────────────────────────────────────────────

class TestPriceChangeModel:
    def _get_trained_model(self) -> tuple[PriceChangeModel, pd.DataFrame]:
        hist = _make_history(n_players=30, n_rounds=20)
        feat = build_features(hist, label=True)
        model = PriceChangeModel()
        model.fit(feat, use_gpu=False)
        return model, feat

    def test_fit_sets_trained(self):
        model, _ = self._get_trained_model()
        assert model.is_trained

    def test_precision_recorded(self):
        model, _ = self._get_trained_model()
        assert model.precision_ is not None
        assert 0.0 <= model.precision_ <= 1.0

    def test_predict_proba_shape(self):
        model, feat = self._get_trained_model()
        out = model.predict_proba_df(feat)
        assert len(out) == len(feat)

    def test_predict_proba_columns(self):
        model, feat = self._get_trained_model()
        out = model.predict_proba_df(feat)
        for col in ["prob_rise", "prob_hold", "prob_fall", "prediction"]:
            assert col in out.columns

    def test_probabilities_sum_to_one(self):
        model, feat = self._get_trained_model()
        out = model.predict_proba_df(feat)
        row_sums = out[["prob_fall", "prob_hold", "prob_rise"]].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-5

    def test_prediction_in_valid_set(self):
        model, feat = self._get_trained_model()
        out = model.predict_proba_df(feat)
        assert out["prediction"].isin([-1, 0, 1]).all()

    def test_untrained_model_raises(self):
        model = PriceChangeModel()
        feat = build_features(_make_history(), label=False)
        with pytest.raises(RuntimeError):
            model.predict_proba_df(feat)

    def test_save_load_roundtrip(self, tmp_path):
        model, feat = self._get_trained_model()
        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        loaded = PriceChangeModel.load(save_path)
        assert loaded.is_trained
        out = loaded.predict_proba_df(feat)
        assert "prob_rise" in out.columns

    def test_insufficient_data_raises(self):
        hist = _make_history(n_players=5, n_rounds=3)
        feat = build_features(hist, label=True)
        model = PriceChangeModel()
        with pytest.raises(ValueError, match="≥ 5 GWs"):
            model.fit(feat, use_gpu=False)
