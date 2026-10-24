"""Tests for screener/ui/portfolio_risk_view.py component using Streamlit AppTest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.portfolio import PORTFOLIO, WATCHLIST, PList


def test_portfolio_risk_view_synthetic():
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 300.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},  # $3000 (30%) -> >20%
        {"ticker": "MSFT", "date": "2025-01-01", "price": 250.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},  # $2500 (25%)
        {"ticker": "JNJ", "date": "2025-01-01", "price": 100.0, "quantity": 15, "side": "buy", "fees": 0, "note": ""},   # $1500 (15%)
        {"ticker": "PG", "date": "2025-01-01", "price": 150.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},    # $1500 (15%)
        {"ticker": "XOM", "date": "2025-01-01", "price": 150.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},   # $1500 (15%)
    ]
    plist = PList("test_p", PORTFOLIO, trades)

    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 300.0, "company": "Apple Inc", "sector": "Electronic Technology", "beta": 1.2},
        {"ticker": "MSFT", "price": 250.0, "company": "Microsoft Corp", "sector": "Electronic Technology", "beta": 1.0},
        {"ticker": "JNJ", "price": 100.0, "company": "Johnson & Johnson", "sector": "Health Technology", "beta": 0.6},
        {"ticker": "PG", "price": 150.0, "company": "Procter & Gamble", "sector": "Consumer Non-Durables", "beta": 0.5},
        {"ticker": "XOM", "price": 150.0, "company": "Exxon Mobil", "sector": "Energy Minerals", "beta": np.nan},
    ])

    def script(p, u):
        from screener.ui.portfolio_risk_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Metrics check
    assert len(at.metric) >= 3
    positions_metric = [m for m in at.metric if "פוזיציות" in m.label]
    assert len(positions_metric) > 0
    assert positions_metric[0].value == "5"

    # Beta coverage check (XOM missing beta -> 85% coverage < 100%)
    beta_metric = [m for m in at.metric if "בטא" in m.label]
    assert len(beta_metric) > 0
    assert "85%" in str(beta_metric[0].delta)

    # Concentration warning check (AAPL is 30% > 20%)
    assert len(at.warning) > 0
    assert any("AAPL" in w.value and "30%" in w.value for w in at.warning)

    # Chart and table existence check
    chart = at.get("plotly_chart")
    assert len(chart) > 0

    tbl = at.get("dataframe")
    assert len(tbl) > 0


def test_portfolio_risk_view_empty_portfolio():
    plist = PList("empty_p", PORTFOLIO, [])
    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 300.0, "company": "Apple Inc", "sector": "Electronic Technology", "beta": 1.2},
    ])

    def script(p, u):
        from screener.ui.portfolio_risk_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Should render empty portfolio note, no metrics or charts
    assert len(at.metric) == 0
    assert len(at.get("plotly_chart")) == 0


def test_portfolio_risk_view_watchlist():
    items = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 150.0, "note": ""},
    ]
    plist = PList("test_w", WATCHLIST, items)
    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 300.0, "company": "Apple Inc", "sector": "Electronic Technology", "beta": 1.2},
    ])

    def script(p, u):
        from screener.ui.portfolio_risk_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Watchlist returns early without any output
    assert len(at.metric) == 0
    assert len(at.warning) == 0
    assert len(at.get("plotly_chart")) == 0


def test_portfolio_risk_view_exception_handled():
    trades = [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 300.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ]
    plist = PList("test_p", PORTFOLIO, trades)
    universe = pd.DataFrame([
        {"ticker": "AAPL", "price": 300.0, "company": "Apple Inc", "sector": "Electronic Technology", "beta": 1.2},
    ])

    def script(p, u):
        import screener.portfolio_risk as pr

        def failing_exposure(*args, **kwargs):
            raise RuntimeError("Simulated exposure error")

        pr.exposure = failing_exposure

        from screener.ui.portfolio_risk_view import render
        render(p, u)

    at = AppTest.from_function(script, args=(plist, universe)).run()

    # Exception should be caught internally by render, so at.exception is empty
    assert not at.exception
