"""AlphaPilot Streamlit dashboard.

Run with:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

REPORTS = Path("reports")
TRADES_LOG = Path("paper_trades.csv")

st.set_page_config(page_title="AlphaPilot", layout="wide")
st.title("\U0001f680 AlphaPilot \u2014 AI Paper Trading Dashboard")
st.caption("\u26a0\ufe0f Paper trading only. Not financial advice.")

# ── Ticker selector ───────────────────────────────────────────────────────────
available = sorted(p.name for p in REPORTS.iterdir() if p.is_dir()) if REPORTS.exists() else []
if not available:
    st.error("No per-ticker reports found. Run `python train_backtest.py --ticker SPY` first.")
    st.stop()

ticker = st.selectbox("Ticker", available, index=0)
ticker_dir = REPORTS / ticker

# ── Metrics ──────────────────────────────────────────────────────────────────
metrics_path = ticker_dir / "metrics.json"
if metrics_path.exists():
    with metrics_path.open(encoding="utf-8") as fh:
        metrics = json.load(fh)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy CAGR", f"{metrics['strategy_cagr']:.1%}")
    col2.metric("Strategy Sharpe", f"{metrics['strategy_sharpe']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['strategy_max_drawdown']:.1%}")
    col4.metric("Win Rate", f"{metrics['win_rate']:.1%}")

    st.subheader("Full Metrics")
    st.json(metrics)
else:
    st.warning(f"No metrics.json found for {ticker}.")

# ── Equity Curve ─────────────────────────────────────────────────────────────
equity_path = ticker_dir / "equity_curve.parquet"
if equity_path.exists():
    equity_df = pd.read_parquet(equity_path)
    st.subheader("Equity Curve vs Buy & Hold")
    st.line_chart(equity_df)
else:
    st.info("Equity curve not available yet.")

# ── Latest signal per ticker ──────────────────────────────────────────────────
st.subheader("Latest Signal per Ticker")
if TRADES_LOG.exists():
    trades = pd.read_csv(TRADES_LOG)
    # Latest row per ticker (by signal_date then by row order)
    latest_per_ticker = (
        trades.sort_values("signal_date").groupby("ticker").last().reset_index()
    )
    for _, row in latest_per_ticker.iterrows():
        color = "\U0001f7e2" if row["signal"] == "LONG" else "\U0001f534"
        st.metric(
            f"{row['ticker']} \u2014 {row['signal_date']}",
            f"{color} {row['signal']}",
            f"P(up) = {row['prob_up']:.4f}",
        )

    st.subheader("Full Trade Log")
    st.dataframe(trades.tail(30), width="stretch")
else:
    st.info("No paper trades yet \u2014 run `python run_daily.py` first.")
