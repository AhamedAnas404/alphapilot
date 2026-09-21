"""Tests for backtest cost logic and metrics math."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphapilot.backtest import run_backtest
from alphapilot.metrics import cagr, max_drawdown, sharpe

# ── Metrics math ─────────────────────────────────────────────────────────────

def test_cagr_flat():
    """Flat equity curve → 0% CAGR."""
    eq = pd.Series([1.0] * 252)
    assert abs(cagr(eq)) < 1e-9


def test_cagr_double():
    """Equity doubling over exactly 252 days → ~100% CAGR."""
    eq = pd.Series(np.linspace(1.0, 2.0, 252))
    assert abs(cagr(eq) - 1.0) < 0.01


def test_sharpe_zero_vol():
    """Zero-volatility returns → Sharpe = 0 (not inf/nan)."""
    # All same value → std = 0
    ret_flat = pd.Series([0.0] * 100)
    assert sharpe(ret_flat) == 0.0


def test_max_drawdown_no_drawdown():
    """Monotonically increasing equity → drawdown = 0."""
    eq = pd.Series(np.linspace(1.0, 2.0, 100))
    assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known():
    """50% drop then recovery → max drawdown = -0.5."""
    eq = pd.Series([1.0, 0.5, 1.0])
    assert max_drawdown(eq) == pytest.approx(-0.5, abs=1e-9)


# ── Backtest cost logic ───────────────────────────────────────────────────────

def _make_inputs(n: int = 300):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=dates)
    # Always predict > threshold → always long
    prob_up = pd.Series(0.6, index=dates)
    return close, prob_up


def test_always_long_no_turnover_after_first_day(strategy_config):
    """If signal never changes, turnover should be ~0 after day 1."""
    close, prob_up = _make_inputs()
    _, metrics = run_backtest(close, prob_up, strategy_config, {})
    # Only 1 trade (entry on day 1), so turnover should be very low
    assert metrics["turnover_pct"] < 5.0


def test_costs_reduce_returns(strategy_config):
    """Strategy with costs must underperform zero-cost version on same signal."""
    close, prob_up = _make_inputs()
    _, metrics_with_cost = run_backtest(close, prob_up, strategy_config, {})

    zero_cost_cfg = {**strategy_config, "transaction_cost": 0.0, "slippage": 0.0}
    _, metrics_no_cost = run_backtest(close, prob_up, zero_cost_cfg, {})

    assert metrics_with_cost["strategy_cagr"] <= metrics_no_cost["strategy_cagr"]


def test_all_cash_signal(strategy_config):
    """If all predictions are below threshold, strategy stays in cash → ~0 return."""
    close, _ = _make_inputs()
    prob_up = pd.Series(0.3, index=close.index)  # always below 0.52
    _, metrics = run_backtest(close, prob_up, strategy_config, {})
    # Cash position → equity curve flat → CAGR ≈ 0
    assert abs(metrics["strategy_cagr"]) < 0.01


def test_win_rate_hand_computed():
    """Hand-computed example: 3 long days, 2 winners → win_rate = 2/3.

    Layout (zero costs for clarity):
      day 0: prob=0.6 → signal=LONG, position[0]=0 (no prior signal yet)
      day 1: prob=0.6 → signal=LONG, position[1]=1, ret=+1%  WIN
      day 2: prob=0.6 → signal=LONG, position[2]=1, ret=-1%  LOSS
      day 3: prob=0.6 → signal=LONG, position[3]=1, ret=+2%  WIN
      day 4: prob=0.6 → signal=LONG, position[4]=1, ret=+1%  WIN  (4 long days total)
    Expected win_rate = 3/4 = 0.75
    """
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    # close prices that produce the daily returns above
    close = pd.Series([100.0, 101.0, 99.99, 101.99, 103.01], index=dates)
    prob_up = pd.Series([0.6, 0.6, 0.6, 0.6, 0.6], index=dates)
    zero_cost = {"prob_threshold": 0.52, "transaction_cost": 0.0, "slippage": 0.0}
    _, metrics = run_backtest(close, prob_up, zero_cost, {})

    # position = signal.shift(1): day0→0, day1→1, day2→1, day3→1, day4→1
    # strat_ret on long days: day1=+1%, day2≈-1%, day3≈+2%, day4≈+1%
    # wins (ret>0): days 1,3,4 → 3 wins out of 4 long days
    assert metrics["win_rate"] == pytest.approx(0.75, abs=0.01)


def test_oos_start_end_present(strategy_config):
    """oos_start and oos_end must be ISO date strings matching the prediction index."""
    close, prob_up = _make_inputs()
    _, metrics = run_backtest(close, prob_up, strategy_config, {})
    assert "oos_start" in metrics
    assert "oos_end" in metrics
    assert metrics["oos_start"] == close.index[0].date().isoformat()
    assert metrics["oos_end"] == close.index[-1].date().isoformat()


def test_oos_perfect_predictor():
    """A perfect predictor must give oos_accuracy == 1.0 and oos_auc == 1.0.

    Prices alternate up/down so next-day direction is known exactly.
    prob_up = 0.9 when next day is up, 0.1 when next day is down.
    The OOS block must align prob_up[t] with close[t+1] > close[t],
    not with the move into day t.
    """
    n = 20
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    # Alternating: up on odd indices, down on even indices
    prices = [100.0]
    for i in range(1, n):
        prices.append(prices[-1] * (1.01 if i % 2 == 1 else 0.99))
    close = pd.Series(prices, index=dates)

    # Perfect signal: prob_up[t] = 0.9 iff close[t+1] > close[t]
    prob_up = pd.Series(index=dates, dtype=float)
    for i in range(n - 1):
        prob_up.iloc[i] = 0.9 if prices[i + 1] > prices[i] else 0.1
    prob_up.iloc[-1] = 0.5  # last row: no next day, value doesn't matter for OOS

    zero_cost = {"prob_threshold": 0.52, "transaction_cost": 0.0, "slippage": 0.0}
    _, metrics = run_backtest(close, prob_up, zero_cost, {})

    assert metrics["oos_accuracy"] == pytest.approx(1.0, abs=1e-9), (
        f"Expected oos_accuracy=1.0, got {metrics['oos_accuracy']}"
    )
    assert metrics["oos_auc"] == pytest.approx(1.0, abs=1e-9), (
        f"Expected oos_auc=1.0, got {metrics['oos_auc']}"
    )
