"""Tests for screener/ui/earnings_window_view.py component using Streamlit AppTest.
"""
from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def test_render_none_and_empty():
    def script_none():
        from screener.ui.earnings_window_view import render
        render(None, None)

    at_none = AppTest.from_function(script_none).run()
    assert not at_none.exception
    assert any("אין דוחות קרובים" in str(html.value) for html in at_none.html)

    def script_empty():
        import pandas as pd
        from screener.ui.earnings_window_view import render
        render(pd.DataFrame(), {})

    at_empty = AppTest.from_function(script_empty).run()
    assert not at_empty.exception
    assert any("אין דוחות קרובים" in str(html.value) for html in at_empty.html)


def test_render_held_tickers_within_7_days():
    df = pd.DataFrame([
        {"ticker": "AAPL", "days_to_earnings": 3, "held": True, "market_cap": 3e12, "sector": "Technology Services"},
        {"ticker": "NVDA", "days_to_earnings": 5, "held": True, "market_cap": 2e12, "sector": "Electronic Technology"},
        {"ticker": "MSFT", "days_to_earnings": 10, "held": False, "market_cap": 2.8e12, "sector": "Technology Services"},
        {"ticker": "AMZN", "days_to_earnings": 20, "held": True, "market_cap": 1.8e12, "sector": "Retail Trade"},
    ])
    buckets = {
        "today": [],
        "this_week": ["AAPL", "NVDA"],
        "next_week": ["MSFT"],
        "later": ["AMZN"],
    }

    def script(df_arg, buckets_arg):
        from screener.ui.earnings_window_view import render
        render(df_arg, buckets_arg)

    at = AppTest.from_function(script, args=(df, buckets)).run()
    assert not at.exception

    # Subheader check
    assert any("חלון דוחות כספיים קרובים" in sub.value for sub in at.subheader)

    # Metric cards check
    metrics = at.metric
    m_today = [m for m in metrics if "היום" in m.label][0]
    m_this_week = [m for m in metrics if "השבוע" in m.label][0]
    m_next_week = [m for m in metrics if "שבוע הבא" in m.label][0]
    m_later = [m for m in metrics if "מאוחר יותר" in m.label][0]

    assert m_today.value == "0"
    assert m_this_week.value == "2"
    assert m_next_week.value == "1"
    assert m_later.value == "1"

    # Warning callout check
    warnings = at.warning
    assert len(warnings) > 0
    assert "AAPL" in warnings[0].value
    assert "NVDA" in warnings[0].value
    assert "מניות בתיק שמדווחות ב-7 הימים הקרובים" in warnings[0].value

    # Dataframe present
    tables = at.get("dataframe")
    assert len(tables) > 0


def test_render_no_held_tickers():
    df = pd.DataFrame([
        {"ticker": "GOOGL", "days_to_earnings": 4, "held": False, "sector": "Technology Services"},
        {"ticker": "META", "days_to_earnings": 12, "held": False, "sector": "Technology Services"},
    ])
    buckets = {
        "today": [],
        "this_week": ["GOOGL"],
        "next_week": ["META"],
        "later": [],
    }

    def script(df_arg, buckets_arg):
        from screener.ui.earnings_window_view import render
        render(df_arg, buckets_arg)

    at = AppTest.from_function(script, args=(df, buckets)).run()
    assert not at.exception

    metrics = at.metric
    m_this_week = [m for m in metrics if "השבוע" in m.label][0]
    assert m_this_week.value == "1"

    # No warnings should be present when no held tickers within 7 days
    assert len(at.warning) == 0


def test_render_held_tickers_outside_7_days():
    df = pd.DataFrame([
        {"ticker": "TSLA", "days_to_earnings": 15, "held": True, "sector": "Consumer Durables"},
    ])
    buckets = {
        "today": [],
        "this_week": [],
        "next_week": [],
        "later": ["TSLA"],
    }

    def script(df_arg, buckets_arg):
        from screener.ui.earnings_window_view import render
        render(df_arg, buckets_arg)

    at = AppTest.from_function(script, args=(df, buckets)).run()
    assert not at.exception

    # Held ticker is reporting in 15 days, so no warning callout
    assert len(at.warning) == 0


def test_render_error_handling():
    df = pd.DataFrame([{"ticker": "AAPL", "days_to_earnings": 2, "held": True}])

    def script(df_arg):
        from screener.ui import earnings_window_view

        # Force exception by monkeypatching st in module
        old_st = earnings_window_view.st
        try:
            earnings_window_view.st = None
            earnings_window_view.render(df_arg)
        finally:
            earnings_window_view.st = old_st

    at = AppTest.from_function(script, args=(df,)).run()
    assert not at.exception
    assert any("שגיאה בהצגת" in str(html.value) for html in at.html)
