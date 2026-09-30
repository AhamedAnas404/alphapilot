"""AlphaPilot dashboard.

Run with:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

REPORTS = Path("reports")
TRADES_LOG = Path("paper_trades.csv")

# Muted trading palette — not pure red/green
_POS = "#16a34a"   # muted green
_NEG = "#dc2626"   # muted red
_NEU = "#374151"   # neutral dark-grey

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AlphaPilot", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AlphaPilot")
    st.caption("AI paper-trading research pipeline")
    st.divider()

    available = (
        sorted(p.name for p in REPORTS.iterdir() if p.is_dir())
        if REPORTS.exists()
        else []
    )
    if not available:
        st.warning("No reports found. Run `python train_backtest.py --ticker SPY` first.")
        st.stop()

    ticker = st.selectbox("Ticker", available, index=0)
    st.divider()
    st.caption("Paper trading only. Not financial advice.")

ticker_dir = REPORTS / ticker

# ── Load data ─────────────────────────────────────────────────────────────────
metrics_path = ticker_dir / "metrics.json"
equity_path = ticker_dir / "equity_curve.parquet"

metrics: dict | None = None
if metrics_path.exists():
    with metrics_path.open(encoding="utf-8") as fh:
        metrics = json.load(fh)

equity_df: pd.DataFrame | None = None
if equity_path.exists():
    equity_df = pd.read_parquet(equity_path)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("AlphaPilot")
st.caption(
    "Walk-forward LightGBM market-timing research. "
    "All results are out-of-sample. Paper trading only."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_perf, tab_signals, tab_method = st.tabs(
    ["Overview", "Performance", "Signals", "Method"]
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _fmt2(v: float) -> str:
    return f"{v:+.2f}"


def _color(v: float) -> str:
    """Return muted green for positive, muted red for negative, neutral for zero."""
    if v > 0:
        return _POS
    if v < 0:
        return _NEG
    return _NEU


def _colored(label: str, v: float, fmt_fn=_pct) -> str:
    """Return an HTML span with muted color based on sign of v."""
    color = _color(v)
    return f'<span style="color:{color};font-weight:600">{fmt_fn(v)}</span>'


def _colored_caption(label: str, v: float, fmt_fn=_pct) -> str:
    color = _color(v)
    return f'{label}: <span style="color:{color}">{fmt_fn(v)}</span>'


def _style_signed(val: str) -> str:
    """Pandas Styler callback: color a pre-formatted string by its leading sign."""
    s = str(val).strip()
    if s.startswith("+"):
        return f"color: {_POS}; font-weight: 600"
    if s.startswith("-"):
        return f"color: {_NEG}; font-weight: 600"
    return f"color: {_NEU}"


def _build_sparkline(df: pd.DataFrame, n: int = 90) -> alt.Chart:
    """Thin 2-line sparkline of the last n rows, no axes or labels."""
    tail = df.tail(n).reset_index()
    tail.columns = ["date", "Strategy (net)", "Buy & hold"]
    melted = tail.melt("date", var_name="Series", value_name="v")
    return (
        alt.Chart(melted)
        .mark_line(strokeWidth=1.4)
        .encode(
            x=alt.X("date:T", axis=None),
            y=alt.Y("v:Q", axis=None, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Strategy (net)", "Buy & hold"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
        )
        .properties(height=80, title="")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


def _build_equity_chart(df: pd.DataFrame, log_scale: bool) -> alt.VConcatChart:
    """Altair layered chart: equity lines + drawdown area beneath."""
    plot = df.reset_index()
    plot.columns = ["date", "Strategy (net)", "Buy & hold"]
    melted = plot.melt("date", var_name="Series", value_name="Growth of 1.0")

    y_scale = alt.Scale(type="log") if log_scale else alt.Scale(type="linear")
    equity_chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Growth of 1.0:Q", scale=y_scale, title="Growth of 1.0"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Strategy (net)", "Buy & hold"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
                legend=alt.Legend(orient="top-left"),
            ),
            tooltip=["date:T", "Series:N", alt.Tooltip("Growth of 1.0:Q", format=".4f")],
        )
        .properties(height=300)
    )

    strat = plot[["date", "Strategy (net)"]].copy()
    strat["peak"] = strat["Strategy (net)"].cummax()
    strat["Drawdown"] = (strat["Strategy (net)"] - strat["peak"]) / strat["peak"]

    dd_chart = (
        alt.Chart(strat)
        .mark_area(color="#d62728", opacity=0.4)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Drawdown:Q", title="Drawdown", scale=alt.Scale(domainMax=0)),
            tooltip=["date:T", alt.Tooltip("Drawdown:Q", format=".2%")],
        )
        .properties(height=120)
    )

    return alt.vconcat(equity_chart, dd_chart).resolve_scale(x="shared")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if metrics is None:
        st.info(
            f"No metrics found for {ticker}. "
            "Run `python train_backtest.py --ticker " + ticker + "` first."
        )
    else:
        # ── Bordered KPI cards ────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)

        with c1:
            with st.container(border=True):
                st.markdown("**CAGR**")
                st.markdown(
                    _colored("CAGR", metrics["strategy_cagr"]),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _colored_caption("Buy & hold", metrics["bh_cagr"]),
                    unsafe_allow_html=True,
                )

        with c2:
            with st.container(border=True):
                st.markdown("**Sharpe ratio**")
                st.markdown(
                    _colored("Sharpe", metrics["strategy_sharpe"], _fmt2),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _colored_caption("Buy & hold", metrics["bh_sharpe"], _fmt2),
                    unsafe_allow_html=True,
                )

        with c3:
            with st.container(border=True):
                st.markdown("**Max drawdown**")
                # drawdown is always <= 0; color by sign (negative = red)
                st.markdown(
                    _colored("Drawdown", metrics["strategy_max_drawdown"]),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _colored_caption("Buy & hold", metrics["bh_max_drawdown"]),
                    unsafe_allow_html=True,
                )

        # ── Sparkline ─────────────────────────────────────────────────────────
        if equity_df is not None and len(equity_df) > 0:
            st.caption("Last 90 trading days — see Performance tab for full chart.")
            spark = _build_sparkline(equity_df, n=90)
            st.altair_chart(spark, width="stretch")

        st.divider()

        # ── Verdict paragraph ─────────────────────────────────────────────────
        n = metrics["n_days"]
        cost_drag = metrics["gross_cagr"] - metrics["strategy_cagr"]
        auc = metrics["oos_auc"]
        se = math.sqrt(1.0 / (3 * n)) if n > 0 else 0.0
        auc_note = (
            "within two standard errors of 0.5 (no detectable edge)"
            if abs(auc - 0.5) < 2 * se
            else "more than two standard errors from 0.5"
        )
        st.markdown(
            f"**{ticker} OOS summary ({metrics['oos_start']} to {metrics['oos_end']}).**  "
            f"The net strategy returned {_pct(metrics['strategy_cagr'])} per year versus "
            f"{_pct(metrics['bh_cagr'])} for buy-and-hold. "
            f"Transaction costs reduced the gross return by approximately "
            f"{cost_drag * 100:.1f} percentage points per year "
            f"(gross {_pct(metrics['gross_cagr'])}, net {_pct(metrics['strategy_cagr'])}). "
            f"The OOS AUC is {auc:.4f}, which is {auc_note}."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    if metrics is None:
        st.info(f"No metrics found for {ticker}.")
    else:
        primary = pd.DataFrame(
            {
                "Strategy (net)": [
                    _pct(metrics["strategy_cagr"]),
                    _fmt2(metrics["strategy_sharpe"]),
                    _pct(metrics["strategy_max_drawdown"]),
                ],
                "Strategy (before costs)": [
                    _pct(metrics["gross_cagr"]),
                    _fmt2(metrics["gross_sharpe"]),
                    _pct(metrics["gross_max_drawdown"]),
                ],
                "Buy & hold": [
                    _pct(metrics["bh_cagr"]),
                    _fmt2(metrics["bh_sharpe"]),
                    _pct(metrics["bh_max_drawdown"]),
                ],
            },
            index=["CAGR", "Sharpe ratio", "Max drawdown"],
        )
        styled = primary.style.map(_style_signed)
        st.subheader("Return summary")
        st.dataframe(styled, width="stretch")
        st.caption(
            "**Sharpe ratio**: annualised return divided by annualised volatility; "
            "higher is better, >1 is generally considered good.  "
            "**Max drawdown**: largest peak-to-trough loss; closer to 0% is better.  "
            "**OOS AUC**: area under the ROC curve on held-out data; "
            "0.5 means the model predicts no better than a coin flip."
        )

        with st.expander("More metrics"):
            secondary = pd.DataFrame(
                {
                    "Value": [
                        _pct(metrics["win_rate"]),
                        f"{metrics['turnover_pct']:.1f}%",
                        str(metrics["total_trades"]),
                        f"{metrics['exposure_pct']:.1f}%",
                        f"{metrics['avg_holding_days']:.1f} days",
                        f"{metrics['oos_accuracy']:.4f}",
                        f"{metrics['oos_auc']:.4f}",
                        f"{metrics['oos_start']} to {metrics['oos_end']}",
                    ]
                },
                index=[
                    "Win rate",
                    "Turnover",
                    "Total trades",
                    "Exposure",
                    "Avg holding period",
                    "OOS accuracy",
                    "OOS AUC",
                    "OOS period",
                ],
            )
            st.dataframe(secondary, width="stretch")

    # Equity chart — always rendered if data present, regardless of metrics
    if equity_df is not None and len(equity_df) > 0:
        st.subheader("Equity curve")
        log_scale = st.toggle("Log scale", value=False, key="log_toggle")
        chart = _build_equity_chart(equity_df, log_scale)
        st.altair_chart(chart, width="stretch")
    elif equity_df is None:
        st.info(
            f"No equity curve found for {ticker}. "
            "Run `python train_backtest.py --ticker " + ticker + "` first."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    if not TRADES_LOG.exists():
        st.info("No paper trades yet. Run `python run_daily.py` first.")
    else:
        trades = pd.read_csv(TRADES_LOG)
        latest_per_ticker = (
            trades.sort_values("signal_date").groupby("ticker").last().reset_index()
        )

        cols = st.columns(max(len(latest_per_ticker), 1))
        for col, (_, row) in zip(cols, latest_per_ticker.iterrows(), strict=False):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{row['ticker']}**")
                    st.markdown(f"Signal: **{row['signal']}**")
                    st.markdown(f"P(up): {row['prob_up']:.4f}")
                    st.markdown(f"Threshold: {row['threshold']}")
                    st.markdown(f"Signal date: {row['signal_date']}")

        st.divider()

        log = trades.copy()
        log = log.sort_values("signal_date", ascending=False)
        log = log.rename(
            columns={
                "ticker": "Ticker",
                "signal_date": "Signal date",
                "prob_up": "P(up)",
                "signal": "Signal",
                "threshold": "Threshold",
                "date": "Run date",
            }
        )
        display_cols = ["Ticker", "Signal date", "P(up)", "Signal", "Threshold"]
        st.dataframe(log[display_cols].head(50), hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — METHOD
# ══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown(
        """
### How it works

**Data**: Daily OHLCV prices downloaded via yfinance and cached locally.

**Features**: 1/5/10/21-day returns, 10/21-day realised volatility,
RSI-14, MACD histogram, 10/50-day MA ratios, volume z-score.

**Model**: LightGBM binary classifier trained in an expanding walk-forward
loop. The model is retrained every time the training window grows by one
retraining step. No data from the test period ever enters training.

**Signal**: Predict probability that tomorrow's close is higher than today's.
Go long if probability > threshold (0.52); otherwise hold cash.

**Costs**: 10 bps transaction cost + 5 bps slippage per side, applied on
every position change.

**Evaluation**: All metrics are computed on the out-of-sample period only.
In-sample results are never reported.

### Limitations

1. LightGBM can fit noise; walk-forward evaluation prevents reporting in-sample results.
2. Only currently-traded tickers are used (survivorship bias).
3. Transaction costs are approximate; real slippage depends on order size and timing.
4. No regime detection — the model treats bull and bear markets identically.
5. Single-asset; no portfolio diversification or position sizing.
6. Technical features only; no fundamentals, news, or macro data.
7. Daily bars; intraday dynamics are ignored.
"""
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AlphaPilot is for educational and research purposes only. "
    "It is not financial advice and must not be used to make real investment decisions."
)
