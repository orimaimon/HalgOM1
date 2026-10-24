import datetime as dt
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from screener.portfolio import BUY, PORTFOLIO, WATCHLIST, PList


def _script_render(plist, monkeypatch_mode=None):
    import pandas as pd
    import numpy as np
    import datetime as dt

    from screener import portfolio_history as ph
    from screener.ui import portfolio_history_view

    def _make_history(records):
        if not records:
            return pd.DataFrame(columns=ph.prices.COLUMNS)
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        for col in ph.prices.COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 if col in ("dividends", "splits") else np.nan
        return df[ph.prices.COLUMNS]

    if monkeypatch_mode == "normal":
        ph.prices.split_factor = lambda t, d: 1.0
        hist_aapl = _make_history([
            {"date": "2025-01-01", "close": 100.0, "adj_close": 100.0, "dividends": 0.0},
            {"date": "2025-01-02", "close": 110.0, "adj_close": 110.0, "dividends": 2.0},
            {"date": "2025-01-03", "close": 120.0, "adj_close": 120.0, "dividends": 0.0},
        ])
        hist_spy = _make_history([
            {"date": "2025-01-01", "close": 500.0, "adj_close": 500.0, "dividends": 0.0},
            {"date": "2025-01-02", "close": 510.0, "adj_close": 510.0, "dividends": 0.0},
            {"date": "2025-01-03", "close": 520.0, "adj_close": 520.0, "dividends": 0.0},
        ])
        hist_qqq = _make_history([
            {"date": "2025-01-01", "close": 400.0, "adj_close": 400.0, "dividends": 0.0},
            {"date": "2025-01-02", "close": 410.0, "adj_close": 410.0, "dividends": 0.0},
            {"date": "2025-01-03", "close": 420.0, "adj_close": 420.0, "dividends": 0.0},
        ])
        def mock_history(t):
            if t == "AAPL":
                return hist_aapl
            if t == "SPY":
                return hist_spy
            if t == "QQQ":
                return hist_qqq
            return pd.DataFrame(columns=ph.prices.COLUMNS)
        ph.prices.history = mock_history

    elif monkeypatch_mode == "no_history":
        ph.prices.split_factor = lambda t, d: 1.0
        ph.prices.history = lambda t: pd.DataFrame(columns=ph.prices.COLUMNS)

    elif monkeypatch_mode == "raising":
        def mock_vs(plist, end=None):
            raise RuntimeError("Backend failure in value_series")
        ph.value_series = mock_vs

    portfolio_history_view.render(plist)


def test_synthetic_portfolio_rendering():
    plist = PList("MyPF", PORTFOLIO, [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0},
    ])
    at = AppTest.from_function(_script_render, args=(plist, "normal"))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception
    # Button ph_update exists
    assert any(b.key == "ph_update" for b in at.button)
    # Plotly chart element exists (accessed via at.get("plotly_chart"))
    assert len(at.get("plotly_chart")) == 1
    # Check notes exist (summary line text)
    notes_html = " ".join([element.value for element in at.html])
    assert "התיק:" in notes_html or "Portfolio:" in notes_html
    assert "%" in notes_html


def test_no_history_note():
    plist = PList("EmptyPF", PORTFOLIO, [
        {"ticker": "UNK", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0},
    ])
    at = AppTest.from_function(_script_render, args=(plist, "no_history"))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception
    assert len(at.get("plotly_chart")) == 0
    notes_html = " ".join([element.value for element in at.html])
    assert "עדיין אין היסטוריה" in notes_html


def test_button_calls_update_prices_on_click():
    def _script_update(plist, clicked):
        import streamlit as st
        from screener import portfolio_history as ph
        from screener.ui import portfolio_history_view

        calls = []
        def mock_update(lists, today=None):
            calls.append(lists)
            return (["AAPL"], [])

        ph.update_prices = mock_update
        st.session_state["calls"] = calls

        if clicked:
            st.session_state["ph_update"] = True

        portfolio_history_view.render(plist)

    plist = PList("MyPF", PORTFOLIO, [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0},
    ])

    # Render without click
    at1 = AppTest.from_function(_script_update, args=(plist, False))
    at1.run()
    if at1.exception:
        raise at1.exception[0]
    assert len(at1.session_state["calls"]) == 0

    # Click button
    at2 = AppTest.from_function(_script_update, args=(plist, False))
    at2.run()
    at2.button(key="ph_update").click().run()
    if at2.exception:
        raise at2.exception[0]
    assert len(at2.session_state["calls"]) == 1
    assert "MyPF" in at2.session_state["calls"][0]


def test_watchlist_no_output():
    plist = PList("MyWatchlist", WATCHLIST, [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "note": ""},
    ])
    at = AppTest.from_function(_script_render, args=(plist, "normal"))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception
    assert len(at.button) == 0
    assert len(at.get("plotly_chart")) == 0


def test_value_series_raising_error_path():
    plist = PList("ErrorPF", PORTFOLIO, [
        {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0},
    ])
    at = AppTest.from_function(_script_render, args=(plist, "raising"))
    at.run()
    assert not at.exception
    notes_html = " ".join([element.value for element in at.html])
    assert "לא ניתן להציג" in notes_html
