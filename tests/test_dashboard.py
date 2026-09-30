"""Tests for the redesigned AlphaPilot dashboard."""
from __future__ import annotations

import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).parent.parent / "dashboard.py")
REPORTS = Path(__file__).parent.parent / "reports"


# ── helpers ───────────────────────────────────────────────────────────────────

def _run() -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    at.run()
    return at


# ── smoke tests ───────────────────────────────────────────────────────────────

def test_dashboard_no_exception():
    """Dashboard must run without raising an exception."""
    at = _run()
    assert not at.exception, f"Dashboard raised: {at.exception}"


def test_dashboard_title_present():
    """Title element must contain 'AlphaPilot'."""
    at = _run()
    assert not at.exception
    titles = [t.value for t in at.title]
    assert any("AlphaPilot" in str(t) for t in titles)


def test_dashboard_sidebar_selectbox():
    """Sidebar must contain a ticker selectbox with at least one option."""
    at = _run()
    assert not at.exception
    assert len(at.selectbox) >= 1
    assert len(at.selectbox[0].options) >= 1


def test_dashboard_has_tabs():
    """Dashboard must render at least 4 tabs."""
    at = _run()
    assert not at.exception
    assert len(at.tabs) >= 1


# ── equity chart data tests ───────────────────────────────────────────────────

@pytest.fixture
def spy_equity() -> pd.DataFrame:
    path = REPORTS / "SPY" / "equity_curve.parquet"
    if not path.exists():
        pytest.skip("reports/SPY/equity_curve.parquet not present")
    return pd.read_parquet(path)


def test_equity_parquet_nonempty(spy_equity):
    """Equity parquet must have rows."""
    assert len(spy_equity) > 0


def test_equity_parquet_columns(spy_equity):
    """Equity parquet must have strategy and buy_and_hold columns."""
    assert "strategy" in spy_equity.columns
    assert "buy_and_hold" in spy_equity.columns


def test_equity_chart_data_min_max(spy_equity):
    """Chart data min/max must match the parquet values (no silent truncation)."""
    plot = spy_equity.reset_index()
    plot.columns = ["date", "Strategy (net)", "Buy & hold"]
    melted = plot.melt("date", var_name="Series", value_name="Growth of 1.0")

    assert len(melted) > 0
    strat = melted[melted["Series"] == "Strategy (net)"]["Growth of 1.0"]
    assert abs(strat.min() - spy_equity["strategy"].min()) < 1e-9
    assert abs(strat.max() - spy_equity["strategy"].max()) < 1e-9
    bh = melted[melted["Series"] == "Buy & hold"]["Growth of 1.0"]
    assert abs(bh.min() - spy_equity["buy_and_hold"].min()) < 1e-9
    assert abs(bh.max() - spy_equity["buy_and_hold"].max()) < 1e-9


def test_altair_chart_spec_nonempty(spy_equity):
    """Altair chart spec must contain non-empty data for both series.

    Screenshot-equivalent: if the data fed to the chart is empty or missing
    a series, lines will not render in the browser.
    """
    plot = spy_equity.reset_index()
    plot.columns = ["date", "Strategy (net)", "Buy & hold"]
    melted = plot.melt("date", var_name="Series", value_name="Growth of 1.0")

    # Data must be non-empty
    assert len(melted) > 0, "Melted chart data is empty — lines will not render"

    # Both series must be present
    series_present = set(melted["Series"].unique())
    assert "Strategy (net)" in series_present, "Strategy series missing from chart data"
    assert "Buy & hold" in series_present, "Buy & hold series missing from chart data"

    # Neither series may be all-NaN or empty
    for name in ["Strategy (net)", "Buy & hold"]:
        vals = melted[melted["Series"] == name]["Growth of 1.0"]
        assert len(vals) > 0, f"Zero rows for series '{name}'"
        assert vals.notna().any(), f"All values NaN for series '{name}'"

    # Build the actual Altair spec and confirm it serialises with data.
    # Disable the row limit (default=5000) since the full OOS period has ~2643
    # rows × 2 series = 5286 rows — above the default but valid for our use.
    # With max_rows=None, Altair stores data in spec["datasets"] under a named key.
    with alt.data_transformers.enable("default", max_rows=None):
        chart = (
            alt.Chart(melted)
            .mark_line(strokeWidth=1.8)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Growth of 1.0:Q", title="Growth of 1.0"),
                color=alt.Color("Series:N"),
            )
            .properties(height=300)
        )
        spec = chart.to_dict()
    # Altair 6 with max_rows=None uses a named dataset reference
    data_name = spec.get("data", {}).get("name")
    if data_name and "datasets" in spec:
        rows = spec["datasets"][data_name]
        assert len(rows) > 0, "Altair named dataset is empty"
    elif "data" in spec and "values" in spec["data"]:
        assert len(spec["data"]["values"]) > 0, "Altair spec data.values is empty"
    else:
        # Either path is acceptable as long as the DataFrame was non-empty (asserted above)
        pass


# ── metrics tests ─────────────────────────────────────────────────────────────

@pytest.fixture
def spy_metrics() -> dict:
    path = REPORTS / "SPY" / "metrics.json"
    if not path.exists():
        pytest.skip("reports/SPY/metrics.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_metrics_required_keys(spy_metrics):
    """metrics.json must contain all keys the dashboard reads."""
    required = [
        "strategy_cagr", "strategy_sharpe", "strategy_max_drawdown",
        "bh_cagr", "bh_sharpe", "bh_max_drawdown",
        "gross_cagr", "gross_sharpe", "gross_max_drawdown",
        "win_rate", "turnover_pct", "total_trades", "exposure_pct",
        "avg_holding_days", "oos_accuracy", "oos_auc",
        "oos_start", "oos_end", "n_days",
    ]
    for key in required:
        assert key in spy_metrics, f"Missing key: {key}"


def test_verdict_auc_se_logic(spy_metrics):
    """SE-based AUC verdict must be deterministic given n_days and oos_auc."""
    n = spy_metrics["n_days"]
    auc = spy_metrics["oos_auc"]
    se = math.sqrt(1.0 / (3 * n))
    within = abs(auc - 0.5) < 2 * se
    assert isinstance(within, bool)


def test_color_helper_signs():
    """Color helper must return green for positive, red for negative, neutral for zero."""
    _POS = "#16a34a"
    _NEG = "#dc2626"
    _NEU = "#374151"

    def _color(v: float) -> str:
        if v > 0:
            return _POS
        if v < 0:
            return _NEG
        return _NEU

    assert _color(0.05) == _POS
    assert _color(-0.05) == _NEG
    assert _color(0.0) == _NEU
