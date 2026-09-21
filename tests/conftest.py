"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """500 rows of synthetic OHLCV data with a deterministic random seed."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def feature_config() -> dict:
    return {
        "returns_windows": [1, 5],
        "vol_windows": [10],
        "rsi_window": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ma_windows": [10, 50],
        "vol_zscore_window": 20,
    }


@pytest.fixture
def model_config() -> dict:
    return {
        "min_train_size": 100,
        "n_estimators": 50,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "random_state": 42,
    }


@pytest.fixture
def strategy_config() -> dict:
    return {
        "prob_threshold": 0.52,
        "transaction_cost": 0.001,
        "slippage": 0.0005,
    }
