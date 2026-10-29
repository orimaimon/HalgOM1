"""Tests for screener/ui/screen_diff_view.py."""
import pytest
from streamlit.testing.v1 import AppTest


def test_screen_diff_view_normal():
    result = {
        "entered": ["AAPL", "NVDA"],
        "left": ["MSFT"],
        "stayed": ["AMZN", "GOOGL"],
        "n_old": 3,
        "n_new": 4,
    }

    def script(res):
        from screener.ui.screen_diff_view import render
        render(res, label="תוצאות השוואה")

    at = AppTest.from_function(script, args=(result,)).run()
    assert not at.exception
    assert any("תוצאות השוואה" in s.value for s in at.subheader)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["נכנסו"] == "2"
    assert metrics["יצאו"] == "1"
    assert metrics["נשארו"] == "2"
    assert metrics["גודל קודם"] == "3"
    assert metrics["גודל חדש"] == "4"
    assert any("AAPL, NVDA" in m.value for m in at.markdown)
    assert any("MSFT" in m.value for m in at.markdown)


def test_screen_diff_view_no_change():
    result = {
        "entered": [],
        "left": [],
        "stayed": ["AAPL", "MSFT"],
        "n_old": 2,
        "n_new": 2,
    }

    def script(res):
        from screener.ui.screen_diff_view import render
        render(res)

    at = AppTest.from_function(script, args=(result,)).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["נכנסו"] == "0"
    assert metrics["יצאו"] == "0"
    assert metrics["נשארו"] == "2"
    assert any("אין שינוי" in i.value for i in at.info)


def test_screen_diff_view_none_input():
    def script(res):
        from screener.ui.screen_diff_view import render
        render(res)

    at = AppTest.from_function(script, args=(None,)).run()
    assert not at.exception
    assert any("אין נתוני השוואה זמינים" in i.value for i in at.info)


def test_screen_diff_view_missing_keys():
    result = {
        "entered": ["AAPL"],
        "left": ["MSFT"],
    }

    def script(res):
        from screener.ui.screen_diff_view import render
        render(res)

    at = AppTest.from_function(script, args=(result,)).run()
    assert not at.exception
    assert any("נתוני השוואה חסרים או שגויים" in i.value for i in at.info)


def test_screen_diff_view_invalid_types():
    result = {
        "entered": "AAPL",
        "left": None,
        "stayed": [],
        "n_old": "3",
        "n_new": 4,
    }

    def script(res):
        from screener.ui.screen_diff_view import render
        render(res)

    at = AppTest.from_function(script, args=(result,)).run()
    assert not at.exception
    assert any("נתוני השוואה חסרים או שגויים" in i.value for i in at.info)


def test_screen_diff_view_very_long_list():
    entered_tickers = [f"TICK{i}" for i in range(300)]
    left_tickers = [f"OLD{i}" for i in range(150)]
    result = {
        "entered": entered_tickers,
        "left": left_tickers,
        "stayed": ["KEEP1", "KEEP2"],
        "n_old": 152,
        "n_new": 302,
    }

    def script(res):
        from screener.ui.screen_diff_view import render
        render(res)

    at = AppTest.from_function(script, args=(result,)).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["נכנסו"] == "300"
    assert metrics["יצאו"] == "150"
    assert metrics["נשארו"] == "2"


def test_screen_diff_view_exception_handling():
    result = {
        "entered": ["AAPL"],
        "left": [],
        "stayed": [],
        "n_old": 0,
        "n_new": 1,
    }

    def script(res):
        import streamlit as st
        orig_columns = st.columns
        try:
            def bad_columns(*args, **kwargs):
                raise RuntimeError("UI error")
            st.columns = bad_columns
            from screener.ui.screen_diff_view import render
            render(res)
        finally:
            st.columns = orig_columns

    at = AppTest.from_function(script, args=(result,)).run()
    assert not at.exception
    assert any("שגיאה" in h.value for h in at.html)
