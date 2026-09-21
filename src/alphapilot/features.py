"""Feature engineering — all features are computed with strict look-ahead safety.

Every rolling/lagged operation uses only past data.  The target label is
created by shifting the forward return *backward* by one row so that row t
carries the label for the move from t→t+1.  Features at row t use only
information available at the close of day t.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual feature builders
# ---------------------------------------------------------------------------

def _returns(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    feats = {}
    for w in windows:
        feats[f"ret_{w}d"] = close.pct_change(w)
    return pd.DataFrame(feats, index=close.index)


def _rolling_vol(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    log_ret = np.log(close / close.shift(1))
    feats = {}
    for w in windows:
        feats[f"vol_{w}d"] = log_ret.rolling(w).std()
    return pd.DataFrame(feats, index=close.index)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{window}")


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line},
        index=close.index,
    )


def _ma_ratios(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    feats = {}
    for w in windows:
        feats[f"ma_ratio_{w}"] = close / close.rolling(w).mean()
    return pd.DataFrame(feats, index=close.index)


def _volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return ((volume - mu) / sigma.replace(0, np.nan)).rename(f"vol_zscore_{window}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, fcfg: dict, drop_unlabeled: bool = True) -> pd.DataFrame:
    """Return feature matrix + binary target for one ticker's OHLCV dataframe.

    Target = 1 if next-day close > today's close, else 0.
    All features are available at end-of-day t (no look-ahead).

    Parameters
    ----------
    drop_unlabeled:
        True  (default, training) — drop the final row whose next-day close is
        unknown, so no fabricated label enters the training set.
        False (inference) — keep the final row; target column will be NaN there.
    """
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    parts = [
        _returns(close, fcfg["returns_windows"]),
        _rolling_vol(close, fcfg["vol_windows"]),
        _rsi(close, fcfg["rsi_window"]).to_frame(),
        _macd(close, fcfg["macd_fast"], fcfg["macd_slow"], fcfg["macd_signal"]),
        _ma_ratios(close, fcfg["ma_windows"]),
        _volume_zscore(volume, fcfg["vol_zscore_window"]).to_frame(),
    ]

    features = pd.concat(parts, axis=1)

    # Target: next-day direction — NaN where next close is unknown (last row).
    # Assign as float first so NaN is representable, cast to int after dropping.
    next_ret = close.pct_change(1).shift(-1)
    features["target"] = np.where(next_ret.isna(), np.nan, (next_ret > 0).astype(float))

    # Drop rows with NaN features; also drop the unlabeled last row when training.
    if drop_unlabeled:
        features = features.dropna()
        features["target"] = features["target"].astype(int)
    else:
        # Drop only feature NaNs; keep the last row even though target is NaN.
        feat_cols = [c for c in features.columns if c != "target"]
        features = features.dropna(subset=feat_cols)

    logger.debug("Built features: %d rows × %d cols (drop_unlabeled=%s)", *features.shape, drop_unlabeled)
    return features
