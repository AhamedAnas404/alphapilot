"""Main pipeline: data → features → walk-forward model → backtest → save results.

Usage:
    python train_backtest.py [--ticker SPY] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from alphapilot.backtest import run_backtest, save_results
from alphapilot.config import load_config
from alphapilot.data import fetch_ticker
from alphapilot.features import build_features
from alphapilot.model import walk_forward_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("train_backtest")


def main(ticker: str | None = None, config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    dcfg, fcfg, mcfg, scfg, bcfg = (
        cfg["data"],
        cfg["features"],
        cfg["model"],
        cfg["strategy"],
        cfg["backtest"],
    )

    ticker = ticker or dcfg["tickers"][0]
    logger.info("=== AlphaPilot pipeline: %s ===", ticker)

    df = fetch_ticker(ticker, dcfg["start"], dcfg.get("end"), Path(dcfg["cache_dir"]))
    feat_df = build_features(df, fcfg)

    logger.info("Feature matrix: %d rows × %d cols", *feat_df.shape)

    wf_result = walk_forward_predict(feat_df, mcfg)

    logger.info("Top-5 features:\n%s", wf_result.feature_importance.head())

    close = df["Close"].squeeze().reindex(feat_df.index)
    equity_df, metrics = run_backtest(close, wf_result.predictions, scfg, bcfg, ticker=ticker)

    logger.info("=== Metrics ===")
    for k, v in metrics.items():
        logger.info("  %-30s %s", k, v)

    save_results(equity_df, metrics, bcfg["reports_dir"])
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaPilot train + backtest")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.ticker, args.config)
