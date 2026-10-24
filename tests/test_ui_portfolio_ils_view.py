import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.data import boi
from screener.portfolio import BUY, PORTFOLIO, WATCHLIST, PList


def test_boi_load_rates_returns_cache(tmp_path, monkeypatch):
    b_dir = tmp_path / "boi"
    b_dir.mkdir(parents=True, exist_ok=True)
    dates = [dt.date(2026, 10, 1), dt.date(2026, 10, 2)]
    s = pd.Series([3.70, 3.65], index=dates)
    boi._atomic_parquet(s, b_dir / "usd_ils.parquet")

    monkeypatch.setattr(boi, "_boi_dir", lambda: b_dir)
    loaded = boi.load_rates()
    assert len(loaded) == 2
    assert loaded.loc[dt.date(2026, 10, 1)] == 3.70


def _script_synthetic(plist_dict, universe_df, rates_series):
    import datetime as dt
    import pandas as pd
    from screener.data import boi, store
    from screener.portfolio import PList
    from screener.ui import portfolio_ils_view

    plist = PList(
        name=plist_dict["name"],
        kind=plist_dict["kind"],
        rows=plist_dict["rows"],
    )
    rates = pd.Series(rates_series) if rates_series is not None else pd.Series(dtype=float)

    boi.load_rates = lambda: rates
    store.market_date = lambda *a, **kw: dt.date(2026, 10, 5)

    portfolio_ils_view.render(plist, universe_df)


def test_render_synthetic_portfolio_and_totals():
    plist_dict = {
        "name": "My Portfolio",
        "kind": PORTFOLIO,
        "rows": [
            {
                "ticker": "AAPL",
                "date": dt.date(2026, 10, 1),
                "price": 100.0,
                "quantity": 10.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            }
        ],
    }
    universe_df = pd.DataFrame({"ticker": ["AAPL"], "price": [110.0]})
    rates_series = {
        dt.date(2026, 10, 1): 3.70,
        dt.date(2026, 10, 5): 3.50,
    }

    at = AppTest.from_function(_script_synthetic, args=(plist_dict, universe_df, rates_series))
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Key button exists
    assert at.button(key="ils_update") is not None
    # Dataframe exists
    assert len(at.dataframe) > 0

    # Stock part (370) + FX part (-220) == P&L (150)
    # Total text formatted in markdown / html note
    html_contents = [h.value for h in at.html]
    total_note = next((h for h in html_contents if "סה״כ: רווח" in h), "")
    assert "סה״כ: רווח" in total_note
    assert "₪150" in total_note
    assert "מזה מניות" in total_note
    assert "₪370" in total_note
    assert "מזה שער הדולר" in total_note
    assert "₪-220" in total_note


def _script_no_rates_and_button(plist_dict, universe_df, should_click):
    import datetime as dt
    import pandas as pd
    from screener.data import boi
    from screener.portfolio import PList
    from screener.ui import portfolio_ils_view

    plist = PList(
        name=plist_dict["name"],
        kind=plist_dict["kind"],
        rows=plist_dict["rows"],
    )

    boi.load_rates = lambda: pd.Series(dtype=float)
    update_calls = []

    def mock_update(start, today=None):
        update_calls.append(start)
        return True

    boi.update = mock_update

    portfolio_ils_view.render(plist, universe_df)


def test_render_no_rates_and_button_click():
    plist_dict = {
        "name": "My Portfolio",
        "kind": PORTFOLIO,
        "rows": [
            {
                "ticker": "AAPL",
                "date": dt.date(2026, 10, 1),
                "price": 100.0,
                "quantity": 10.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            }
        ],
    }
    universe_df = pd.DataFrame({"ticker": ["AAPL"], "price": [110.0]})

    # First run without click: boi.update never called
    at = AppTest.from_function(_script_no_rates_and_button, args=(plist_dict, universe_df, False))
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception
    assert any("עדיין אין שערים שמורים — לחצו על עדכון" in info.value for info in at.info)

    # Click update button
    at.button(key="ils_update").click().run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception


def _script_watchlist_returns_nothing(plist_dict, universe_df):
    from screener.portfolio import PList
    from screener.ui import portfolio_ils_view

    plist = PList(
        name=plist_dict["name"],
        kind=plist_dict["kind"],
        rows=plist_dict["rows"],
    )
    portfolio_ils_view.render(plist, universe_df)


def test_render_watchlist_returns_without_output():
    plist_dict = {
        "name": "My Watchlist",
        "kind": WATCHLIST,
        "rows": [{"ticker": "AAPL", "price": 100.0}],
    }
    universe_df = pd.DataFrame({"ticker": ["AAPL"], "price": [110.0]})

    at = AppTest.from_function(_script_watchlist_returns_nothing, args=(plist_dict, universe_df))
    at.run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception
    assert len(at.button) == 0
    assert len(at.dataframe) == 0


def _script_exception_handling(plist_dict, universe_df):
    from screener import portfolio_ils
    from screener.portfolio import PList
    from screener.ui import portfolio_ils_view

    def boom(*args, **kwargs):
        raise RuntimeError("Calculation error")

    portfolio_ils.ils_positions = boom

    plist = PList(
        name=plist_dict["name"],
        kind=plist_dict["kind"],
        rows=plist_dict["rows"],
    )
    portfolio_ils_view.render(plist, universe_df)


def test_render_exception_handled():
    plist_dict = {
        "name": "My Portfolio",
        "kind": PORTFOLIO,
        "rows": [
            {
                "ticker": "AAPL",
                "date": dt.date(2026, 10, 1),
                "price": 100.0,
                "quantity": 10.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            }
        ],
    }
    universe_df = pd.DataFrame({"ticker": ["AAPL"], "price": [110.0]})

    at = AppTest.from_function(_script_exception_handling, args=(plist_dict, universe_df))
    at.run()

    assert not at.exception
    html_contents = [h.value for h in at.html]
    assert any("אירעה שגיאה בחישוב רווח/הפסד בשקלים" in h for h in html_contents)
