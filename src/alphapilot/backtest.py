"""Backtest engine: converts probability signals into an equity curve with costs."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from alphapilot.metrics import cagr as _cagr
from alphapilot.metrics import max_drawdown as _max_drawdown
from alphapilot.metrics import sharpe as _sharpe

logger = logging.getLogger(__name__)


def run_backtest(
    close: pd.Series,
    prob_up: pd.Series,
    scfg: dict,
    bcfg: dict,
    ticker: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Simulate strategy and return (equity_df, metrics_dict).

    Parameters
    ----------
    close:    daily close prices (full history)
    prob_up:  out-of-sample predicted probability of up move (NaN before min_train)
    scfg:     strategy config section
    bcfg:     backtest config section
    ticker:   ticker symbol stored in metrics for reference
    """
    threshold = scfg["prob_threshold"]
    cost = scfg["transaction_cost"]
    slippage = scfg["slippage"]

    # Align on dates where we have predictions
    valid = prob_up.dropna()
    close_aligned = close.reindex(valid.index)

    # Daily returns of the underlying
    daily_ret = close_aligned.pct_change().fillna(0.0)

    # Signal: 1 = long, 0 = cash  (signal at t applies to return at t+1)
    signal = (valid > threshold).astype(int)

    # Strategy return: signal[t-1] * market_return[t] - costs when signal changes
    position = signal.shift(1).fillna(0)
    traded = position.diff().abs().fillna(0).astype(bool)

    strat_ret = pd.Series(index=valid.index, dtype=float)
    for i, (dt, pos) in enumerate(position.items()):
        mr = daily_ret.iloc[i]
        strat_ret[dt] = pos * mr - (cost + slippage) * float(traded.iloc[i])

    # Equity curves (start at 1.0)
    strat_equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + daily_ret).cumprod()

    equity_df = pd.DataFrame(
        {"strategy": strat_equity, "buy_and_hold": bh_equity},
        index=valid.index,
    )

    metrics = _compute_metrics(strat_ret, daily_ret, strat_equity, bh_equity, position, traded)

    # Gross (zero-cost) pass — same signal, no transaction costs or slippage
    gross_ret = pd.Series(index=valid.index, dtype=float)
    for i, (dt, pos) in enumerate(position.items()):
        gross_ret[dt] = pos * daily_ret.iloc[i]
    gross_equity = (1 + gross_ret).cumprod()
    metrics["gross_cagr"] = round(_cagr(gross_equity), 4)
    metrics["gross_sharpe"] = round(_sharpe(gross_ret), 4)
    metrics["gross_max_drawdown"] = round(_max_drawdown(gross_equity), 4)

    # OOS classification quality
    # prob_up[t] predicts close[t+1] > close[t], so actual label is next-day direction.
    # Drop the last row: it has no next-day close, so the label is undefined.
    actual_dir = (close_aligned.shift(-1) > close_aligned).astype(int)
    actual_dir = actual_dir.iloc[:-1]
    valid_trimmed = valid.iloc[:-1]
    pred_label = (valid_trimmed > threshold).astype(int)
    metrics["oos_accuracy"] = round(float((pred_label == actual_dir).mean()), 4)
    try:
        from sklearn.metrics import roc_auc_score
        metrics["oos_auc"] = round(float(roc_auc_score(actual_dir, valid_trimmed)), 4)
    except Exception:  # noqa: BLE001
        metrics["oos_auc"] = None

    # Exposure and holding-period stats
    n = len(position)
    metrics["exposure_pct"] = round(float(position.sum() / n * 100), 2)
    # avg consecutive days in position per trade
    runs = (position != position.shift()).cumsum()
    long_runs = position[position == 1].groupby(runs[position == 1]).count()
    metrics["avg_holding_days"] = round(float(long_runs.mean()), 2) if len(long_runs) else 0.0

    metrics["ticker"] = ticker
    metrics["oos_start"] = valid.index[0].date().isoformat()
    metrics["oos_end"] = valid.index[-1].date().isoformat()

    return equity_df, metrics


def _compute_metrics(
    strat_ret: pd.Series,
    bh_ret: pd.Series,
    strat_eq: pd.Series,
    bh_eq: pd.Series,
    position: pd.Series,
    traded: pd.Series,
) -> dict:
    n = len(strat_ret)
    wins = (strat_ret[position == 1] > 0).sum()
    total_trades = int((position.diff().abs() > 0).sum())
    trade_days = int(position.sum())

    return {
        "strategy_cagr": round(_cagr(strat_eq), 4),
        "strategy_sharpe": round(_sharpe(strat_ret), 4),
        "strategy_max_drawdown": round(_max_drawdown(strat_eq), 4),
        "bh_cagr": round(_cagr(bh_eq), 4),
        "bh_sharpe": round(_sharpe(bh_ret), 4),
        "bh_max_drawdown": round(_max_drawdown(bh_eq), 4),
        "win_rate": round(float(wins / trade_days) if trade_days > 0 else 0.0, 4),
        "turnover_pct": round(float(traded.sum() / n * 100), 2),
        "total_trades": total_trades,
        "n_days": n,
    }


def save_results(
    equity_df: pd.DataFrame,
    metrics: dict,
    reports_dir: str | Path,
) -> None:
    """Persist equity curve (parquet) and metrics (JSON) to reports/<ticker>/."""
    import json

    ticker = metrics.get("ticker") or "unknown"
    rdir = Path(reports_dir) / ticker
    rdir.mkdir(parents=True, exist_ok=True)

    equity_df.to_parquet(rdir / "equity_curve.parquet")

    with (rdir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved equity curve and metrics to %s", rdir)
    _plot_equity(equity_df, rdir)


def _plot_equity(equity_df: pd.DataFrame, rdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        equity_df["strategy"].plot(ax=ax, label="AlphaPilot Strategy", linewidth=1.5)
        equity_df["buy_and_hold"].plot(ax=ax, label="Buy & Hold", linewidth=1.5, linestyle="--")
        ax.set_title("AlphaPilot — Equity Curve vs Buy & Hold")
        ax.set_ylabel("Portfolio Value (normalised to 1)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(rdir / "equity_curve.png", dpi=150)
        plt.close(fig)
        logger.info("Saved equity_curve.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save plot: %s", exc)
