"""Tests for scripts/update_readme.py and run_daily dedup logic."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from update_readme import END_MARKER, START_MARKER, build_block, update  # noqa: E402

# Make repo root importable for run_daily
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from run_daily import _existing_keys  # noqa: E402


@pytest.fixture
def sample_metrics() -> dict:
    return {
        "strategy_cagr": -0.08,
        "strategy_sharpe": -0.50,
        "strategy_max_drawdown": -0.60,
        "bh_cagr": 0.15,
        "bh_sharpe": 0.90,
        "bh_max_drawdown": -0.34,
        "gross_cagr": 0.04,
        "gross_sharpe": 0.35,
        "gross_max_drawdown": -0.38,
        "win_rate": 0.51,
        "turnover_pct": 38.0,
        "total_trades": 1000,
        "oos_accuracy": 0.50,
        "oos_auc": 0.49,
        "exposure_pct": 56.0,
        "avg_holding_days": 3.0,
        "oos_start": "2016-01-01",
        "oos_end": "2026-01-01",
        "ticker": "SPY",
    }


def test_build_block_contains_key_fields(sample_metrics):
    block = build_block(sample_metrics, others=[])
    assert "2016-01-01 \u2192 2026-01-01" in block
    assert "SPY" in block
    assert "equity_curve.png" in block
    assert "OOS AUC" in block
    assert "Win Rate" in block


def test_update_readme_replaces_block(tmp_path, sample_metrics):
    # update() now takes (reports_dir, readme_path, headline, others)
    # Set up reports/SPY/metrics.json under tmp_path
    spy_dir = tmp_path / "reports" / "SPY"
    spy_dir.mkdir(parents=True)
    (spy_dir / "metrics.json").write_text(json.dumps(sample_metrics), encoding="utf-8")

    readme_file = tmp_path / "README.md"
    readme_file.write_text(
        f"# Title\n\n{START_MARKER}\nOLD CONTENT\n{END_MARKER}\n\n## Footer\n",
        encoding="utf-8",
    )

    update(tmp_path / "reports", readme_file, "SPY", [])

    result = readme_file.read_text(encoding="utf-8")
    assert "OLD CONTENT" not in result
    assert START_MARKER in result
    assert END_MARKER in result
    assert "2016-01-01" in result
    assert "## Footer" in result  # content outside markers preserved


def test_update_readme_missing_markers_exits(tmp_path, sample_metrics):
    spy_dir = tmp_path / "reports" / "SPY"
    spy_dir.mkdir(parents=True)
    (spy_dir / "metrics.json").write_text(json.dumps(sample_metrics), encoding="utf-8")

    readme_file = tmp_path / "README.md"
    readme_file.write_text("# No markers here\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        update(tmp_path / "reports", readme_file, "SPY", [])


# ── run_daily dedup ───────────────────────────────────────────────────────────

FIELDNAMES = ["date", "signal_date", "ticker", "prob_up", "signal", "threshold"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_existing_keys_empty(tmp_path):
    assert _existing_keys(tmp_path / "missing.csv") == set()


def test_existing_keys_reads_pairs(tmp_path):
    p = tmp_path / "trades.csv"
    _write_csv(p, [
        {"date": "2024-01-02", "signal_date": "2024-01-01", "ticker": "SPY",
         "prob_up": 0.6, "signal": "LONG", "threshold": 0.52},
        {"date": "2024-01-03", "signal_date": "2024-01-02", "ticker": "AAPL",
         "prob_up": 0.4, "signal": "CASH", "threshold": 0.52},
    ])
    keys = _existing_keys(p)
    assert ("2024-01-01", "SPY") in keys
    assert ("2024-01-02", "AAPL") in keys
    assert len(keys) == 2


def test_existing_keys_no_duplicate_written(tmp_path):
    """A row whose (signal_date, ticker) already exists must not be appended."""
    p = tmp_path / "trades.csv"
    existing_row = {
        "date": "2024-01-02", "signal_date": "2024-01-01", "ticker": "SPY",
        "prob_up": 0.6, "signal": "LONG", "threshold": 0.52,
    }
    _write_csv(p, [existing_row])

    keys = _existing_keys(p)
    # Simulate what main() does: skip if key already present
    new_row = {**existing_row, "date": "2024-01-03"}  # same signal_date+ticker
    rows_to_write = []
    if (new_row["signal_date"], new_row["ticker"]) not in keys:
        rows_to_write.append(new_row)

    assert rows_to_write == [], "Duplicate row should have been skipped"
