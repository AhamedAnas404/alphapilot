"""Standalone metric helpers (also used by tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate from a normalised equity curve."""
    years = len(equity) / TRADING_DAYS
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free / TRADING_DAYS
    sigma = excess.std()
    return float(excess.mean() / sigma * np.sqrt(TRADING_DAYS)) if sigma > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())
