"""Replace the RESULTS block in README.md from per-ticker reports/.

Usage:
    python scripts/update_readme.py [--reports reports/] [--readme README.md]
    python scripts/update_readme.py --headline SPY --others AAPL MSFT
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"


def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def load_metrics(reports_dir: Path, ticker: str) -> dict:
    path = reports_dir / ticker / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_block(m: dict, others: list[dict]) -> str:
    oos_period = f"{m['oos_start']} \u2192 {m['oos_end']}"
    cost_drag = round(m["gross_cagr"] - m["strategy_cagr"], 4)
    ticker = m["ticker"]
    lines = [
        f"OOS period: **{oos_period}** (ticker: {ticker})",
        "",
        f"![Equity Curve](reports/{ticker}/equity_curve.png)",
        "",
        f"| Metric | Net Strategy | Gross (zero-cost) | Buy & Hold ({ticker}) |",
        "|---|---|---|---|",
        f"| CAGR | **{_pct(m['strategy_cagr'])}** | {_pct(m['gross_cagr'])} | {_pct(m['bh_cagr'])} |",
        f"| Sharpe | {_fmt(m['strategy_sharpe'])} | {_fmt(m['gross_sharpe'])} | {_fmt(m['bh_sharpe'])} |",
        f"| Max Drawdown | {_pct(m['strategy_max_drawdown'])} | {_pct(m['gross_max_drawdown'])} | {_pct(m['bh_max_drawdown'])} |",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Win Rate | {_pct(m['win_rate'])} |",
        f"| Turnover | {_fmt(m['turnover_pct'])}% |",
        f"| Total Trades | {m['total_trades']} |",
        f"| OOS Accuracy | {_fmt(m['oos_accuracy'], 4)} |",
        f"| OOS AUC | {_fmt(m['oos_auc'], 4)} |",
        f"| Exposure | {_fmt(m['exposure_pct'])}% |",
        f"| Avg Holding Days | {_fmt(m['avg_holding_days'])} |",
        "",
        "> The net strategy **underperforms** buy-and-hold. This is the honest result.",
        f"> Transaction costs removed ~{_pct(cost_drag)} CAGR (gross {_pct(m['gross_cagr'])} \u2192 net {_pct(m['strategy_cagr'])}).",
        f"> OOS AUC = {_fmt(m['oos_auc'], 4)} (near 0.5) \u2014 no detectable predictive edge.",
    ]

    if others:
        lines += [
            "",
            "### Other tickers (same model, same parameters)",
            "",
            "| Ticker | Net CAGR | Sharpe | OOS AUC |",
            "|---|---|---|---|",
        ]
        for o in others:
            lines.append(
                f"| {o['ticker']} | {_pct(o['strategy_cagr'])} | {_fmt(o['strategy_sharpe'])} | {_fmt(o['oos_auc'], 4)} |"
            )

    return "\n".join(lines)


def update(reports_dir: Path, readme_path: Path, headline: str, others: list[str]) -> None:
    m = load_metrics(reports_dir, headline)
    other_metrics = [load_metrics(reports_dir, t) for t in others]

    text = readme_path.read_text(encoding="utf-8")
    block = build_block(m, other_metrics)
    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{block}\n{END_MARKER}"
    new_text, n = pattern.subn(replacement, text)
    if n == 0:
        print("ERROR: markers not found in README", file=sys.stderr)
        sys.exit(1)

    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {readme_path} (headline={headline}, others={others})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports/")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--headline", default="SPY")
    parser.add_argument("--others", nargs="*", default=[])
    args = parser.parse_args()
    update(Path(args.reports), Path(args.readme), args.headline, args.others)


if __name__ == "__main__":
    main()
