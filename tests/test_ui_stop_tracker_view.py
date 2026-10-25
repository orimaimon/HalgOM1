"""Tests for Portfolio Trailing-Stop & Target Tracker UI component (screener/ui/stop_tracker_view.py)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener import alerts, portfolio
from screener.ui import stop_tracker_view


def script_render_stop_tracker(plist, universe, alerts_path):
    from pathlib import Path
    from screener import alerts
    from screener.ui.stop_tracker_view import render

    alerts.ALERTS_PATH = Path(alerts_path)
    render(plist, universe)


def script_render_error(plist, universe):
    from screener import portfolio
    from screener.ui.stop_tracker_view import render

    orig_positions = portfolio.positions
    def bad_positions(*args, **kwargs):
        raise RuntimeError("Position calculation failure")

    portfolio.positions = bad_positions
    try:
        render(plist, universe)
    finally:
        portfolio.positions = orig_positions


def test_stop_tracker_render_positions_and_warnings(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    # Trades for 3 tickers
    trades = pd.DataFrame([
        {"ticker": "AAPL", "date": dt.date(2025, 1, 1), "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "MSFT", "date": dt.date(2025, 1, 1), "price": 200.0, "quantity": 5, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "NVDA", "date": dt.date(2025, 1, 1), "price": 150.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ])
    plist = portfolio.PList(name="Test Portfolio", kind=portfolio.PORTFOLIO, rows=portfolio._records(trades))

    # ATR mult default = 2.0
    # Stop price = max(avg_cost, price, high_1m) - 2.0 * atr
    # distance_pct = (price - stop_price) / price * 100
    #
    # 1. AAPL: avg_cost=100, high_1m=120, price=90, atr=10 -> ref_high=120 -> stop_price = 120 - 20 = 100
    #    price (90) <= stop_price (100) -> status "חצה סטופ"
    #
    # 2. MSFT: avg_cost=200, high_1m=210, price=200, atr=5 -> ref_high=210 -> stop_price = 210 - 10 = 200
    #    price=200, stop_price=198? Let's tune:
    #    Suppose ref_high=200, atr=1 -> stop_price = 198 -> distance_pct = (200 - 198)/200 * 100 = 1.0%
    #    0 < distance_pct <= 1.5% -> status "קרוב לסטופ"
    #
    # 3. NVDA: avg_cost=150, high_1m=160, price=160, atr=2 -> ref_high=160 -> stop_price = 156
    #    distance_pct = (160 - 156)/160 * 100 = 2.5% > 1.5% -> status "תקין"
    universe = pd.DataFrame([
        {"ticker": "AAPL", "company": "Apple", "price": 90.0, "atr": 10.0, "high_1m": 120.0},
        {"ticker": "MSFT", "company": "Microsoft", "price": 200.0, "atr": 1.0, "high_1m": 200.0},
        {"ticker": "NVDA", "company": "Nvidia", "price": 160.0, "atr": 2.0, "high_1m": 160.0},
    ])

    at = AppTest.from_function(script_render_stop_tracker, args=(plist, universe, str(alerts_file)))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Check that 2 warnings appear for AAPL and MSFT
    assert len(at.warning) == 2
    assert "AAPL" in at.warning[0].value
    assert "MSFT" in at.warning[1].value

    # Check dataframe rendered
    assert len(at.dataframe) == 1


def test_quick_alert_creation(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    trades = pd.DataFrame([
        {"ticker": "AAPL", "date": dt.date(2025, 1, 1), "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ])
    plist = portfolio.PList(name="Test Portfolio", kind=portfolio.PORTFOLIO, rows=portfolio._records(trades))
    universe = pd.DataFrame([
        {"ticker": "AAPL", "company": "Apple", "price": 120.0, "atr": 5.0, "high_1m": 120.0},
    ])

    at = AppTest.from_function(script_render_stop_tracker, args=(plist, universe, str(alerts_file)))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Click set alert button
    at.button(key="st_set_alert").click().run()
    if at.exception:
        raise at.exception[0]

    loaded = alerts.load(alerts_file)
    assert len(loaded) == 1
    a = list(loaded.values())[0]
    assert a.ticker == "AAPL"
    assert a.kind == "price_below"
    # ref_high = 120, atr = 5, mult = 2 -> stop_price = 110.0
    assert a.value == 110.0
    assert a.note == "סטופ נגרר ATR"


def test_watchlist_or_empty_portfolio(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    universe = pd.DataFrame([{"ticker": "AAPL", "price": 100.0}])

    # 1. Watchlist
    wl_plist = portfolio.PList(name="My Watchlist", kind=portfolio.WATCHLIST, rows=[{"ticker": "AAPL"}])
    at_wl = AppTest.from_function(script_render_stop_tracker, args=(wl_plist, universe, str(alerts_file)))
    at_wl.run()
    if at_wl.exception:
        raise at_wl.exception[0]
    assert len(at_wl.warning) == 0

    # 2. Empty Portfolio
    empty_plist = portfolio.PList(name="Empty Portfolio", kind=portfolio.PORTFOLIO, rows=[])
    at_emp = AppTest.from_function(script_render_stop_tracker, args=(empty_plist, universe, str(alerts_file)))
    at_emp.run()
    if at_emp.exception:
        raise at_emp.exception[0]
    assert len(at_emp.warning) == 0


def test_error_path(tmp_path):
    trades = pd.DataFrame([
        {"ticker": "AAPL", "date": dt.date(2025, 1, 1), "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ])
    plist = portfolio.PList(name="Test Portfolio", kind=portfolio.PORTFOLIO, rows=portfolio._records(trades))
    universe = pd.DataFrame([{"ticker": "AAPL", "price": 100.0}])

    at_err = AppTest.from_function(script_render_error, args=(plist, universe))
    at_err.run()
    assert not at_err.exception
