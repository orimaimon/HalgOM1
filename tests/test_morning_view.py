"""Tests for screener.ui.morning_view module."""
import datetime as dt
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener import portfolio
from screener.ui.morning_view import render_morning_dashboard


def make_sample_universe() -> pd.DataFrame:
    today_date = dt.date.today()
    earn_date = today_date + dt.timedelta(days=3)
    return pd.DataFrame([
        {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "price": 180.0,
            "change": 2.5,
            "earnings_date": earn_date.isoformat(),
            "new": True,
        },
        {
            "ticker": "MSFT",
            "company": "Microsoft Corp.",
            "price": 400.0,
            "change": -1.2,
            "earnings_date": (today_date + dt.timedelta(days=20)).isoformat(),
        },
        {
            "ticker": "NVDA",
            "company": "NVIDIA Corp.",
            "price": 120.0,
            "change": 5.8,
            "days_to_earnings": 5,
            "new": True,
        },
    ])


def make_sample_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [500.25, 430.50, 15.20],
            "change": [1.2, -0.8, -3.5],
        },
        index=["SPY", "QQQ", "VIX"],
    )


def make_sample_portfolios() -> dict:
    p1 = portfolio.PList("Main Portfolio", portfolio.PORTFOLIO, [
        {"ticker": "AAPL", "date": "2026-01-01", "price": 150.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "NVDA", "date": "2026-01-01", "price": 100.0, "quantity": 20, "side": "buy", "fees": 0, "note": ""},
    ])
    w1 = portfolio.PList("Tech Watchlist", portfolio.WATCHLIST, [
        {"ticker": "MSFT", "date": "2026-01-01", "price": 380.0, "quantity": 0, "side": "buy", "fees": 0, "note": ""},
    ])
    return {
        "Main Portfolio": p1,
        "Tech Watchlist": w1,
    }


def test_render_morning_dashboard_with_realistic_data():
    universe_df = make_sample_universe()
    market_df = make_sample_market()
    portfolios_data = make_sample_portfolios()

    def script(univ_dict, market_dict, pf_dict):
        import pandas as pd
        import streamlit as st
        from screener.portfolio import PList, PORTFOLIO, WATCHLIST
        from screener.ui.morning_view import render_morning_dashboard

        univ = pd.DataFrame(univ_dict)
        mkt = pd.DataFrame(market_dict) if market_dict else None

        pfs = {}
        if pf_dict:
            for k, v in pf_dict.items():
                pfs[k] = PList(name=v["name"], kind=v["kind"], rows=v["rows"])

        render_morning_dashboard(univ, mkt, pfs)

    pf_serializable = {
        k: {"name": v.name, "kind": v.kind, "rows": v.rows}
        for k, v in portfolios_data.items()
    }

    at = AppTest.from_function(
        script,
        args=(
            universe_df.to_dict("list"),
            market_df.to_dict("dict"),
            pf_serializable,
        ),
    )
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Check that metrics and dataframes rendered
    assert len(at.metric) >= 4  # SPY, QQQ, VIX + New Scan Candidates count
    assert len(at.dataframe) >= 2  # Movers table + Earnings table


def test_render_morning_dashboard_edge_case_empty_portfolio():
    universe_df = make_sample_universe()
    market_df = make_sample_market()

    def script(univ_dict, market_dict):
        import pandas as pd
        from screener.ui.morning_view import render_morning_dashboard

        univ = pd.DataFrame(univ_dict)
        mkt = pd.DataFrame(market_dict)

        render_morning_dashboard(univ, mkt, portfolios_data={})

    at = AppTest.from_function(
        script,
        args=(universe_df.to_dict("list"), market_df.to_dict("dict")),
    )
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Should show info for empty portfolio movers and earnings
    assert any("אין תנועות בולטות" in info.value for info in at.info)
    assert any("אין דוחות כספיים" in info.value for info in at.info)


def test_render_morning_dashboard_edge_case_no_earnings_date():
    universe_df = make_sample_universe().drop(columns=["earnings_date", "days_to_earnings"])
    market_df = make_sample_market()
    portfolios_data = make_sample_portfolios()

    def script(univ_dict, market_dict, pf_dict):
        import pandas as pd
        from screener.portfolio import PList
        from screener.ui.morning_view import render_morning_dashboard

        univ = pd.DataFrame(univ_dict)
        mkt = pd.DataFrame(market_dict)
        pfs = {k: PList(name=v["name"], kind=v["kind"], rows=v["rows"]) for k, v in pf_dict.items()}

        render_morning_dashboard(univ, mkt, pfs)

    pf_serializable = {
        k: {"name": v.name, "kind": v.kind, "rows": v.rows}
        for k, v in portfolios_data.items()
    }

    at = AppTest.from_function(
        script,
        args=(
            universe_df.to_dict("list"),
            market_df.to_dict("dict"),
            pf_serializable,
        ),
    )
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Should render without exception and show "no upcoming earnings" message
    assert any("אין דוחות כספיים" in info.value for info in at.info)


def test_render_morning_dashboard_edge_case_market_df_none():
    universe_df = make_sample_universe()

    def script(univ_dict):
        import pandas as pd
        from screener.ui.morning_view import render_morning_dashboard

        univ = pd.DataFrame(univ_dict)
        render_morning_dashboard(univ, market_df=None, portfolios_data=None)

    at = AppTest.from_function(script, args=(universe_df.to_dict("list"),))
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Market info warning/info note rendered
    assert any("אין נתוני שוק" in info.value for info in at.info)


def test_render_morning_dashboard_all_none():
    def script():
        from screener.ui.morning_view import render_morning_dashboard
        render_morning_dashboard(None, None, None)

    at = AppTest.from_function(script)
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception
