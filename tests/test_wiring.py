"""Tests for UI components wiring into pages (Home, Screener, Groups, Portfolios, Guide)."""
from __future__ import annotations

import datetime as dt
import pandas as pd
import pytest

from screener import portfolio as PF
from screener.ui.common import t
from tests.test_frontend import app  # noqa: F401


def test_home_default_for_new_session(app):
    """Home page 'הבוקר שלי' is the default page for a new session."""
    app.run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception
    assert app.session_state["page"] == "Home"
    assert app.segmented_control(key="page").value == "Home"


def test_screener_page_with_row_selection(app):
    """Screener page renders; selecting a row renders explain_panel and compare_panel with no exception."""
    app.run()
    app.segmented_control(key="page").select("Screener").run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception

    # Verify fast chart review and position sizer expanders are present on Screener results
    expanders = [e.label or "" for e in app.get("expander")]
    assert any("דפדוף מהיר" in lbl or "Fast chart review" in lbl for lbl in expanders)
    assert any("מחשבון גודל" in lbl or "Position sizer" in lbl for lbl in expanders)

    # Select a ticker and re-run to verify no duplicate key or exception
    res = app.session_state["pipeline_res"]
    assert not res.empty
    ticker = str(res.iloc[0]["ticker"])
    app.session_state["selected_ticker"] = ticker
    app.run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception


def test_groups_page_tabs_and_breadth(app):
    """Groups page renders all 4 tabs (Averages, Sector rotation, Track record, Market breadth)."""
    app.run()
    app.segmented_control(key="page").select("Groups").run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception

    # Verify tabs exist
    tabs = app.get("tabs")
    assert len(tabs) == 4


def test_heatmap_page(app):
    """Heatmap page renders with no exception."""
    app.run()
    app.segmented_control(key="page").select("Heatmap").run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception


def test_portfolios_page_tabs_and_alerts(app):
    """Portfolios page renders sample portfolio tabs and alerts expander."""
    # Create sample portfolio
    sample = PF.PList(
        name="Sample",
        kind=PF.PORTFOLIO,
        rows=[
            {"ticker": "AAPL", "date": "2024-01-01", "price": 150.0, "quantity": 10.0, "side": "buy", "fees": 0.0, "note": ""}
        ],
    )
    PF.save(sample)

    app.run()
    app.segmented_control(key="page").select("Portfolios").run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception

    # Verify tabs exist for portfolio (Positions, History, Risk, Trade journal, Shekels)
    tabs = app.get("tabs")
    assert len(tabs) == 5

    # Verify alerts and trailing stops expanders exist
    expanders = [e.label or "" for e in app.get("expander")]
    assert any("התראות" in lbl or "Alerts" in lbl for lbl in expanders)
    assert any("סטופ" in lbl or "Trailing stops" in lbl for lbl in expanders)


def test_portfolios_page_watchlist_triggers(app):
    """Portfolios page renders watchlist triggers expander for a non-empty watchlist."""
    wl = PF.PList(
        name="WatchSample",
        kind=PF.WATCHLIST,
        rows=[{"ticker": "AAPL", "date": "2024-01-01", "price": 150.0, "note": "Breakout"}],
    )
    PF.save(wl)

    app.run()
    app.segmented_control(key="page").select("Portfolios").run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception

    expanders = [e.label or "" for e in app.get("expander")]
    assert any("תוכנית עבודה" in lbl or "Trading playbook" in lbl for lbl in expanders)


def test_guide_page_shows_maintenance_panel(app):
    """Guide page renders guide content and maintenance panel."""
    app.run()
    app.segmented_control(key="page").select("Guide").run()
    if app.exception:
        raise app.exception[0]
    assert not app.exception

    # Maintenance panel buttons/download buttons exist
    buttons = [b.label for b in app.get("button")] + [b.label for b in app.get("download_button")]
    # Verify maintenance panel texts or buttons appear
    assert any("דוח תקלה" in b or "גיבוי" in b or "Backup" in b or "Report" in b for b in buttons)
