"""Tests for screener.ui.home_page component."""
import datetime as dt
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from screener import portfolio
from screener.data.store import Store
from screener.fields import COLUMNS

DAY0 = dt.date(2026, 3, 10)
PREV_DAY = dt.date(2026, 3, 9)


def make_snapshot_df(rows: dict[str, dict]) -> pd.DataFrame:
    tickers = list(rows)
    n = len(tickers)
    data = {k: [10.0] * n for k, c in COLUMNS.items() if c.source == "provider" and c.kind not in ("text", "date")}
    df = pd.DataFrame(data)
    df["ticker"] = tickers
    df["company"] = [f"Co {t}" for t in tickers]
    df["exchange"] = "NASDAQ"
    df["sector"] = "Technology"
    df["industry"] = "Software"
    df["country"] = "United States"
    df["sec_type"] = "stock"
    df["market_cap"], df["avg_volume"] = 1e9, 1e6

    for t, vals in rows.items():
        idx = df[df["ticker"] == t].index[0]
        for col, val in vals.items():
            df.loc[idx, col] = val

    return df


def test_ui_home_page_with_data(tmp_path):
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)

    prev_rows = {
        "AAPL": {"pe": 10.0, "price": 150.0},
        "MSFT": {"pe": 12.0, "price": 300.0},
    }
    make_snapshot_df(prev_rows).to_parquet(store.snapshot_path(PREV_DAY))

    today_date = store.snapshot_path(DAY0)
    today_rows = {
        "AAPL": {"pe": 10.0, "price": 150.0, "change": 5.0, "company": "Apple Inc."},
        "MSFT": {"pe": 12.0, "price": 300.0, "change": -3.0, "company": "Microsoft Corp.", "earnings_date": "2026-03-12"},
        "GOOG": {"pe": 14.0, "price": 105.0, "change": 8.0, "company": "Alphabet Inc."},
    }
    today_univ_df = make_snapshot_df(today_rows)
    today_univ_df.to_parquet(today_date)

    # Portfolio file with held MSFT position
    p1 = portfolio.PList("My Portfolio", portfolio.PORTFOLIO, [
        {"ticker": "MSFT", "date": "2026-01-01", "price": 300.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
    ])
    pf_path = tmp_path / "portfolios.json"
    portfolio._write({"My Portfolio": p1}, pf_path)

    universe_dict = today_univ_df.to_dict("list")

    def script(univ_data, data_dir, fired_msgs):
        import datetime as dt
        from pathlib import Path
        import pandas as pd
        import streamlit as st

        import screener.data.store as store_mod
        import screener.portfolio as portfolio_mod
        from screener.data.store import Store
        from screener.market import Regime
        from screener.ui.home_page import home_page

        store_mod.DATA_DIR = Path(data_dir)
        portfolio_mod.PORTFOLIOS_PATH = Path(data_dir) / "portfolios.json"

        st.cache_data.clear()

        universe = pd.DataFrame(univ_data)
        store = Store(Path(data_dir))
        regime = Regime(label="bull", reasons=["SPY above SMA200"])

        home_page(universe, store, regime, fired_msgs)

    at = AppTest.from_function(script, args=(universe_dict, str(tmp_path), ["MSFT cross above SMA200"]))
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Check key elements exist
    assert at.expander(key="hm_report_expander")
    # Verify tables or elements render
    assert len(at.dataframe) > 0 or len(at.info) > 0


def test_ui_home_page_empty(tmp_path):
    store = Store(tmp_path)
    store.snap_dir.mkdir(parents=True, exist_ok=True)

    empty_univ = make_snapshot_df({})
    universe_dict = empty_univ.to_dict("list")

    def script(univ_data, data_dir):
        from pathlib import Path
        import pandas as pd
        import streamlit as st

        import screener.data.store as store_mod
        import screener.portfolio as portfolio_mod
        from screener.data.store import Store
        from screener.ui.home_page import home_page

        store_mod.DATA_DIR = Path(data_dir)
        portfolio_mod.PORTFOLIOS_PATH = Path(data_dir) / "portfolios.json"

        st.cache_data.clear()

        universe = pd.DataFrame(univ_data)
        store = Store(Path(data_dir))

        home_page(universe, store, None, None)

    at = AppTest.from_function(script, args=(universe_dict, str(tmp_path)))
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Check "אין חדש היום" info notice
    assert any("אין חדש היום" in info.value for info in at.info)


def test_ui_home_page_exception_handling(tmp_path, monkeypatch):
    empty_univ = make_snapshot_df({})
    universe_dict = empty_univ.to_dict("list")

    def script(univ_data, data_dir):
        from pathlib import Path
        import pandas as pd
        import streamlit as st

        import screener.data.store as store_mod
        import screener.home as home_mod
        import screener.portfolio as portfolio_mod
        from screener.data.store import Store
        from screener.ui.home_page import home_page

        store_mod.DATA_DIR = Path(data_dir)
        portfolio_mod.PORTFOLIOS_PATH = Path(data_dir) / "portfolios.json"

        def bad_morning(*args, **kwargs):
            raise RuntimeError("Simulated home.morning failure")

        home_mod.morning = bad_morning

        st.cache_data.clear()

        universe = pd.DataFrame(univ_data)
        store = Store(Path(data_dir))

        home_page(universe, store, None, None)

    at = AppTest.from_function(script, args=(universe_dict, str(tmp_path)))
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Should render note without raising exception
    assert any("לא ניתן" in html.value for html in at.html)
