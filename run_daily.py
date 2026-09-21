"""Daily paper-trading runner.

Usage:
    python run_daily.py [--config config.yaml]

Fetches latest data, generates today's signal for each ticker, and appends
a row to paper_trades.csv.  Designed to be called by a cron / GitHub Actions
scheduled workflow.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alphapilot.config import load_config
from alphapilot.data import fetch_ticker
from alphapilot.features import build_features
from alphapilot.model import walk_forward_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("run_daily")


def _generate_signal(ticker: str, cfg: dict) -> dict:
    """Return today's signal dict for *ticker*."""
    dcfg, fcfg, mcfg, scfg = cfg["data"], cfg["features"], cfg["model"], cfg["strategy"]

    df = fetch_ticker(
        ticker,
        dcfg["start"],
        dcfg.get("end"),
        Path(dcfg["cache_dir"]),
        force=True,  # always refresh on daily run
    )

    feat_df = build_features(df, fcfg, drop_unlabeled=False)
    result = walk_forward_predict(feat_df, mcfg)

    latest_date = result.predictions.dropna().index[-1]
    latest_prob = float(result.predictions.dropna().iloc[-1])
    signal = "LONG" if latest_prob > scfg["prob_threshold"] else "CASH"

    return {
        "date": date.today().isoformat(),
        "signal_date": latest_date.date().isoformat(),
        "ticker": ticker,
        "prob_up": round(latest_prob, 4),
        "signal": signal,
        "threshold": scfg["prob_threshold"],
    }


def _existing_keys(log_file: Path) -> set[tuple[str, str]]:
    """Return set of (signal_date, ticker) pairs already in the CSV."""
    if not log_file.exists():
        return set()
    with log_file.open(newline="") as fh:
        return {
            (row["signal_date"], row["ticker"])
            for row in csv.DictReader(fh)
        }


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    log_file = Path(cfg["paper_trading"]["log_file"])

    fieldnames = ["date", "signal_date", "ticker", "prob_up", "signal", "threshold"]
    write_header = not log_file.exists()
    existing = _existing_keys(log_file)

    rows = []
    for ticker in cfg["data"]["tickers"]:
        try:
            row = _generate_signal(ticker, cfg)
            key = (row["signal_date"], row["ticker"])
            if key in existing:
                logger.info("Skipping duplicate (%s, %s)", *key)
                continue
            rows.append(row)
            logger.info("Signal for %s: %s (prob=%.4f)", ticker, row["signal"], row["prob_up"])
        except Exception:
            logger.exception("Failed to generate signal for %s", ticker)

    with log_file.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logger.info("Appended %d rows to %s", len(rows), log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaPilot daily signal runner")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
