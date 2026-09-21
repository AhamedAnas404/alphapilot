"""Zero look-ahead bias tests — the most critical correctness guarantee."""
from __future__ import annotations

from alphapilot.features import build_features
from alphapilot.model import walk_forward_predict


def test_target_uses_future_return(sample_ohlcv, feature_config):
    """Target at row t must equal sign of close[t+1] - close[t]."""
    feat = build_features(sample_ohlcv, feature_config)
    close = sample_ohlcv["Close"].squeeze()

    for dt, row in feat.iterrows():
        loc = close.index.get_loc(dt)
        if loc + 1 >= len(close):
            continue
        expected = int(close.iloc[loc + 1] > close.iloc[loc])
        assert row["target"] == expected, (
            f"Target mismatch at {dt}: got {row['target']}, expected {expected}"
        )


def test_features_use_only_past_data(sample_ohlcv, feature_config):
    """Features at row t must not change when future rows are removed."""
    feat_full = build_features(sample_ohlcv, feature_config)

    # Take only the first 300 rows of raw data
    cutoff = 300
    feat_partial = build_features(sample_ohlcv.iloc[:cutoff], feature_config)

    # Rows that exist in both should have identical feature values
    common_idx = feat_full.index.intersection(feat_partial.index)
    feat_cols = [c for c in feat_full.columns if c != "target"]

    for col in feat_cols:
        diff = (feat_full.loc[common_idx, col] - feat_partial.loc[common_idx, col]).abs()
        assert diff.max() < 1e-10, f"Feature '{col}' differs — possible look-ahead bias"


def test_walk_forward_no_future_leakage(sample_ohlcv, feature_config, model_config):
    """Predictions at index t must not depend on data after t.

    Runs walk-forward on the full feature set and on a truncated copy
    (first 350 rows).  Every prediction in the overlapping date range must
    be bit-for-bit identical: same training window → same model → same output.
    Any divergence would indicate that future rows influenced a past prediction.
    """
    feat = build_features(sample_ohlcv, feature_config)
    result_full = walk_forward_predict(feat, model_config)

    # Truncate to first 350 rows and re-run
    feat_short = feat.iloc[:350]
    result_short = walk_forward_predict(feat_short, model_config)

    overlap = result_full.predictions.dropna().index.intersection(
        result_short.predictions.dropna().index
    )
    assert len(overlap) > 0, "No overlapping predictions to compare"

    for dt in overlap:
        diff = abs(result_full.predictions[dt] - result_short.predictions[dt])
        assert diff < 1e-9, (
            f"Prediction diverges at {dt}: full={result_full.predictions[dt]:.10f} "
            f"short={result_short.predictions[dt]:.10f} diff={diff:.2e} "
            "\u2014 possible future data leakage"
        )
