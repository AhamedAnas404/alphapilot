"""Download and cache OHLCV data via yfinance."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from alphapilot.config import load_config

logger = logging.getLogger(__name__)


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker}.parquet"


def fetch_ticker(
    ticker: str,
    start: str,
    end: str | None,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for *ticker*, using parquet cache when available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, cache_dir)

    end_str = end or date.today().isoformat()

    if path.exists() and not force:
        df = pd.read_parquet(path)
        logger.info("Loaded %s from cache (%d rows)", ticker, len(df))
        return df

    logger.info("Downloading %s %s → %s", ticker, start, end_str)
    raw = yf.download(ticker, start=start, end=end_str, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance may return MultiIndex columns when downloading a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.to_parquet(path)
    logger.info("Cached %s → %s (%d rows)", ticker, path, len(df))
    return df


def load_all(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Download/load all tickers defined in config and return as {ticker: df}."""
    cfg = cfg or load_config()
    dcfg = cfg["data"]
    cache_dir = Path(dcfg["cache_dir"])
    return {
        t: fetch_ticker(t, dcfg["start"], dcfg.get("end"), cache_dir)
        for t in dcfg["tickers"]
    }
