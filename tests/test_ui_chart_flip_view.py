"""Tests for screener/ui/chart_flip_view.py."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener import portfolio


def make_test_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "price": 220.50,
            "change": 1.25,
            "score": 85.0,
            "atr_pct": 2.10,
            "signals": "BO 52W, Vol 2.5x",
            "tv_symbol": "NASDAQ:AAPL",
        },
        {
            "ticker": "MSFT",
            "company": "Microsoft Corp.",
            "price": 440.00,
            "change": -0.50,
            "score": 90.0,
            "atr_pct": 1.80,
            "signals": "Golden Cross",
            "tv_symbol": "NASDAQ:MSFT",
        },
        {
            "ticker": "NVDA",
            "company": "NVIDIA Corp.",
            "price": 130.00,
            "change": 3.10,
            "score": 95.0,
            "atr_pct": 3.50,
            "signals": "BO 1M",
            "tv_symbol": "NASDAQ:NVDA",
        },
    ])


def test_chart_flip_view_navigation():
    df = make_test_df()

    def script(data):
        from screener.ui.chart_flip_view import render
        render(data)

    at = AppTest.from_function(script, args=(df,)).run()
    assert not at.exception
    assert at.session_state["cf_idx"] == 0

    # Click next -> MSFT (idx == 1)
    at.button(key="cf_next").click().run()
    assert not at.exception
    assert at.session_state["cf_idx"] == 1

    # Click prev -> AAPL (idx == 0)
    at.button(key="cf_prev").click().run()
    assert not at.exception
    assert at.session_state["cf_idx"] == 0


def test_chart_flip_view_empty_dataframe():
    empty_df = pd.DataFrame()

    def script(data):
        from screener.ui.chart_flip_view import render
        render(data)

    at = AppTest.from_function(script, args=(empty_df,)).run()
    assert not at.exception
    assert any("אין מניות להצגה" in h.value for h in at.html)


def test_chart_flip_view_add_to_watchlist(tmp_path, monkeypatch):
    p_path = tmp_path / "portfolios.json"
    monkeypatch.setattr(portfolio, "PORTFOLIOS_PATH", p_path)

    df = make_test_df()

    def script(data):
        from screener.ui.chart_flip_view import render
        render(data)

    at = AppTest.from_function(script, args=(df,)).run()
    assert not at.exception

    # Fill trader note input and click add button
    at.text_input(key="cf_note").input("פריצה יפה").run()
    at.button(key="cf_add_wl").click().run()
    assert not at.exception

    all_lists = portfolio.load_all(p_path)
    assert "מעקב" in all_lists
    wl = all_lists["מעקב"]
    assert len(wl.rows) == 1
    assert wl.rows[0]["ticker"] == "AAPL"
    assert wl.rows[0]["note"] == "פריצה יפה"


def test_chart_flip_view_error_path():
    df = make_test_df()

    def script(data):
        import screener.explain as ex
        orig = ex.describe
        try:
            ex.describe = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("internal crash"))
            from screener.ui.chart_flip_view import render
            render(data)
        finally:
            ex.describe = orig

    at = AppTest.from_function(script, args=(df,)).run()
    assert not at.exception
    assert any("לא ניתן להציג" in h.value for h in at.html)
