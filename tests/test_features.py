"""Tests for feature engineering correctness."""
from __future__ import annotations

import numpy as np

from alphapilot.features import build_features


def test_feature_shape(sample_ohlcv, feature_config):
    """Feature matrix must have fewer rows than raw data (NaN rows dropped)."""
    feat = build_features(sample_ohlcv, feature_config)
    assert len(feat) < len(sample_ohlcv)
    assert "target" in feat.columns


def test_target_is_binary(sample_ohlcv, feature_config):
    feat = build_features(sample_ohlcv, feature_config)
    assert set(feat["target"].unique()).issubset({0, 1})


def test_no_nan_in_features(sample_ohlcv, feature_config):
    feat = build_features(sample_ohlcv, feature_config)
    assert not feat.isnull().any().any(), "Feature matrix contains NaN values"


def test_rsi_bounds(sample_ohlcv, feature_config):
    feat = build_features(sample_ohlcv, feature_config)
    rsi_col = [c for c in feat.columns if c.startswith("rsi_")][0]
    assert feat[rsi_col].between(0, 100).all(), "RSI out of [0, 100] range"


def test_ma_ratio_positive(sample_ohlcv, feature_config):
    feat = build_features(sample_ohlcv, feature_config)
    ma_cols = [c for c in feat.columns if c.startswith("ma_ratio_")]
    for col in ma_cols:
        assert (feat[col] > 0).all(), f"{col} has non-positive values"


def test_no_fabricated_label_on_final_day(sample_ohlcv, feature_config):
    """The last raw row must not appear in the training frame.

    Before the fix, (NaN > 0) evaluated to False, so the last row got
    target=0 and was never removed by dropna().  After the fix, target is
    set to NaN explicitly and dropna() removes it correctly.
    """
    last_raw_date = sample_ohlcv.index[-1]
    feat = build_features(sample_ohlcv, feature_config, drop_unlabeled=True)
    assert last_raw_date not in feat.index, (
        f"Final raw date {last_raw_date} found in training frame — fabricated label present"
    )
    # Also confirm target is integer dtype (no NaN survived)
    assert feat["target"].dtype == np.dtype("int64") or feat["target"].dtype == np.dtype("int32")


def test_inference_keeps_final_row(sample_ohlcv, feature_config):
    """With drop_unlabeled=False the last row is retained for inference."""
    last_raw_date = sample_ohlcv.index[-1]
    feat = build_features(sample_ohlcv, feature_config, drop_unlabeled=False)
    # The last row may or may not be present depending on feature NaNs,
    # but if features are valid it must be present.
    feat_cols = [c for c in feat.columns if c != "target"]
    if not feat[feat_cols].iloc[-1].isna().any():
        assert last_raw_date in feat.index, (
            "Final raw date missing from inference frame — drop_unlabeled=False broken"
        )
        assert np.isnan(feat.loc[last_raw_date, "target"]), (
            "Final row target should be NaN in inference mode"
        )
