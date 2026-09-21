.PHONY: install data train backtest backtest-all dashboard test lint all

PYTHON := python
TICKER  := SPY

install:
	$(PYTHON) -m pip install -e ".[dev]" --quiet

data:
	$(PYTHON) -c "import sys; sys.path.insert(0,'src'); from alphapilot.data import load_all; load_all()"

train:
	$(PYTHON) train_backtest.py --ticker $(TICKER)

backtest: train

backtest-all:
	$(PYTHON) train_backtest.py --ticker SPY
	$(PYTHON) train_backtest.py --ticker AAPL
	$(PYTHON) train_backtest.py --ticker MSFT
	$(PYTHON) scripts/update_readme.py --headline SPY --others AAPL MSFT

dashboard:
	streamlit run dashboard.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check src/ tests/ run_daily.py train_backtest.py dashboard.py scripts/update_readme.py

all: lint test backtest
