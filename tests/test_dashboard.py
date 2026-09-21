"""Smoke-test the Streamlit dashboard with AppTest."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).parent.parent / "dashboard.py")


def test_dashboard_no_exception():
    """Dashboard must run without raising an exception."""
    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    at.run()
    assert not at.exception, f"Dashboard raised: {at.exception}"


def test_dashboard_has_title():
    """Dashboard must render the AlphaPilot title."""
    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    at.run()
    assert not at.exception
    titles = [t.value for t in at.title]
    assert any("AlphaPilot" in str(t) for t in titles)


def test_dashboard_ticker_selectbox():
    """Dashboard must render a ticker selectbox with at least one option."""
    at = AppTest.from_file(DASHBOARD, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 1
    options = at.selectbox[0].options
    assert len(options) >= 1
