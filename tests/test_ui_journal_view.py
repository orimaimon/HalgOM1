"""Tests for screener/ui/journal_view.py component using Streamlit AppTest.
"""
from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.portfolio import PORTFOLIO, WATCHLIST, PList


def test_journal_view_win_and_loss():
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": "buy 1"},
        {"ticker": "AAPL", "date": "2025-01-10", "price": 150.0, "quantity": 5, "side": "sell", "fees": 0, "note": "win 1"},  # cost $500, proceeds $750 -> +$250 (+50%)
        {"ticker": "MSFT", "date": "2025-02-01", "price": 200.0, "quantity": 10, "side": "buy", "fees": 0, "note": "buy 2"},
        {"ticker": "MSFT", "date": "2025-02-15", "price": 180.0, "quantity": 10, "side": "sell", "fees": 0, "note": "loss 1"}, # cost $2000, proceeds $1800 -> -$200 (-10%)
    ]
    plist = PList("test_journal", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "company": "Apple Inc", "sector": "Electronic Technology"},
        {"ticker": "MSFT", "price": 180.0, "company": "Microsoft Corp", "sector": "Technology Services"},
    ])

    def script(p, u):
        from screener.ui.journal_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Top metrics check: 2 closed trades, 50.0% win rate
    assert len(at.metric) >= 4
    trades_metric = [m for m in at.metric if "עסקאות סגורות" in m.label]
    assert len(trades_metric) > 0
    assert trades_metric[0].value == "2"

    winrate_metric = [m for m in at.metric if "אחוז הצלחה" in m.label]
    assert len(winrate_metric) > 0
    assert winrate_metric[0].value == "50.0%"

    pf_metric = [m for m in at.metric if "מכפיל רווח" in m.label]
    assert len(pf_metric) > 0
    assert pf_metric[0].value == "1.25"  # $250 / $200 = 1.25

    rr_metric = [m for m in at.metric if "יחס רווח מול הפסד" in m.label]
    assert len(rr_metric) > 0
    assert rr_metric[0].value == "1.25"  # ($250 / 1) / ($200 / 1) = 1.25

    # Check existence of chart and dataframe elements
    chart = at.get("plotly_chart")
    assert len(chart) > 0

    tbl = at.get("dataframe")
    assert len(tbl) > 0


def test_journal_view_only_buys():
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ]
    plist = PList("test_buys", PORTFOLIO, trades)
    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "company": "Apple Inc", "sector": "Electronic Technology"},
    ])

    def script(p, u):
        from screener.ui.journal_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # No closed trades, so 0 metrics and no chart
    assert len(at.metric) == 0
    assert len(at.get("plotly_chart")) == 0


def test_journal_view_only_winning_sells():
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "AAPL", "date": "2025-01-10", "price": 150.0, "quantity": 10, "side": "sell", "fees": 0, "note": ""},
    ]
    plist = PList("test_wins", PORTFOLIO, trades)
    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "company": "Apple Inc", "sector": "Electronic Technology"},
    ])

    def script(p, u):
        from screener.ui.journal_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Win rate 100.0%, Profit factor "—", Payoff ratio "—"
    winrate_metric = [m for m in at.metric if "אחוז הצלחה" in m.label]
    assert len(winrate_metric) > 0
    assert winrate_metric[0].value == "100.0%"

    pf_metric = [m for m in at.metric if "מכפיל רווח" in m.label]
    assert len(pf_metric) > 0
    assert pf_metric[0].value == "—"

    rr_metric = [m for m in at.metric if "יחס רווח מול הפסד" in m.label]
    assert len(rr_metric) > 0
    assert rr_metric[0].value == "—"


def test_journal_view_watchlist():
    items = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 150.0, "note": ""},
    ]
    plist = PList("test_w", WATCHLIST, items)
    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 300.0, "company": "Apple Inc", "sector": "Electronic Technology"},
    ])

    def script(p, u):
        from screener.ui.journal_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    assert len(at.metric) == 0


def test_journal_view_error_handling(caplog):
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "AAPL", "date": "2025-01-10", "price": 150.0, "quantity": 10, "side": "sell", "fees": 0, "note": ""},
    ]
    plist = PList("test_err", PORTFOLIO, trades)
    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 150.0, "company": "Apple Inc", "sector": "Electronic Technology"},
    ])

    def script(p, u):
        from screener.ui import journal_view
        def failing_compute(*args, **kwargs):
            raise RuntimeError("Simulated journal error")

        journal_view._compute_closed_trades = failing_compute
        journal_view.render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    assert not at.exception
