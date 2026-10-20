"""Front-end behaviour that needs the running app (Streamlit AppTest, offline, synthetic data)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from screener import export, lifecycle
from screener import portfolio as PF
from screener import presets as P
from screener.data.tradingview import clean
from tests.test_screener import ROOT, make_raw


@pytest.fixture
def app(tmp_path, monkeypatch):
    import screener.data.store as store_mod
    import screener.data.tradingview as tv
    monkeypatch.setattr(tv, "MIN_UNIVERSE_ROWS", 100)
    raw = clean(make_raw(300).assign(is_primary=True, currency="USD"))
    raw.attrs["source"] = {"provider": "tradingview", "update_mode": "delayed", "delay_min": 15}
    market = pd.DataFrame({"close": [500.0, 400, 80, 15], "change": 0.5, "sma20": [490.0, 390, 79, 16],
                           "sma50": 480.0, "sma200": [450.0, 350, 78, 17], "adx": [30.0, 20, 15, 10],
                           "di_plus": 25.0, "di_minus": 15.0}, index=["SPY", "QQQ", "HYG", "VIX"])
    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "PRESETS_PATH", tmp_path / "presets.json")
    monkeypatch.setattr(PF, "PORTFOLIOS_PATH", tmp_path / "portfolios.json")
    monkeypatch.setattr(export, "EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(tv, "fetch_universe", lambda: raw)
    monkeypatch.setattr(tv, "fetch_market", lambda: market)
    monkeypatch.setattr(lifecycle, "start_idle_watchdog", lambda: None)
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(str(ROOT / "screener" / "app.py"), default_timeout=60)


def notes(at: AppTest) -> list[str]:
    """Bodies of the st.html blocks (hebrew_note)."""
    return [e.proto.body for e in at.get("html")]


def test_column_config():
    from screener.app import column_config

    assert column_config("ticker")["pinned"] is True
    assert column_config("company")["pinned"] is False

    widths = {
        "ticker": "small", "exchange": "small", "country": "small", "sec_type": "small",
        "company": "medium", "sector": "medium", "industry": "medium",
        "signals": 300, "indexes": 240, "new": 60
    }
    for key, expected_width in widths.items():
        assert column_config(key)["width"] == expected_width

def test_range52_bar():
    from screener.app import range52_bar
    import math

    # Valid range, exactly in the middle
    html = range52_bar(150.0, 100.0, 200.0)
    assert "left:50.0%" in html

    # Missing data or no range: no bar at all (a bar in the middle would be made up)
    assert range52_bar(float('nan'), 100.0, 200.0) == ""
    assert range52_bar(150.0, float('nan'), 200.0) == ""
    assert range52_bar(150.0, 150.0, 150.0) == ""

    # Price above High
    html_above = range52_bar(250.0, 100.0, 200.0)
    assert "left:100.0%" in html_above

    # Price below Low
    html_below = range52_bar(50.0, 100.0, 200.0)
    assert "left:0.0%" in html_below


def test_find_removed_pill():
    from screener.app import find_removed_pill

    assert find_removed_pill(["pe", "market_cap"], ["pe"]) == "market_cap"
    assert find_removed_pill(["pe", "market_cap"], ["market_cap"]) == "pe"
    assert find_removed_pill(["pe"], []) == "pe"
    assert find_removed_pill(["pe"], ["pe"]) is None


def test_active_filters_pills(app):
    app.run()
    app.selectbox(key="f_pe").select("Low (<15)").run()
    assert not app.exception, app.exception

    def pills():
        return next(p for p in app.get("button_group") if (p.key or "").startswith("active_pills_"))

    app.selectbox(key="f_market_cap").select(app.selectbox(key="f_market_cap").options[2]).run()
    # a filter added later is shown selected too (the old single-key widget kept only ['pe'])
    assert len(pills().value) == 2 and pills().value == pills().options

    # clicking the P/E pill removes only that filter; the market-cap pill stays selected
    cap = next(o for o in pills().options if "P/E" not in o)
    pills().set_value([cap]).run()
    assert not app.exception, app.exception
    assert app.session_state["f_pe"] == "Any" and app.session_state["f_market_cap"] != "Any"
    assert pills().value == [cap]


def test_large_text_toggle(app):
    app.run()
    assert not app.exception, app.exception

    # Toggle off by default
    assert not app.session_state["large_text"]
    assert not any("115%" in h for h in notes(app))

    # Enable Large Text
    app.toggle(key="large_text").set_value(True).run()
    assert app.session_state["large_text"]
    assert any("115%" in h for h in notes(app))


def test_data_time_header(app):
    app.run()
    assert not app.exception, app.exception
    assert any("שעת הנתונים:" in c.value for c in app.caption)


def test_empty_results_explain_and_clear_everything(app):
    app.run()
    app.selectbox(key="f_pe").select("Low (<15)").run()
    app.text_input(key="query").input("NO-SUCH-STOCK").run()
    assert not app.exception, app.exception
    assert any("אף מניה לא עומדת" in h for h in notes(app))
    button = app.button(key="clear_all")
    assert button.label == "איפוס כל הפילטרים והחיפוש"
    button.click().run()
    assert not app.exception, app.exception
    assert app.session_state["f_pe"] == "Any" and app.session_state["query"] == ""
    assert not any("אף מניה לא עומדת" in h for h in notes(app))
    assert "clear_all" not in [b.key for b in app.button]


# ── Portfolios page ─────────────────────────────────────────────────────────

def open_portfolios(app: AppTest) -> list[str]:
    """Run the app, go to the Portfolios page; returns some universe tickers."""
    app.run()
    tickers = list(app.session_state["df"]["ticker"][:3])
    app.session_state["page"] = "Portfolios"
    app.run()
    assert not app.exception, app.exception
    return tickers


def test_watchlist_create_add_and_show_in_screener(app):
    a, b, _ = open_portfolios(app)
    assert any("עוד אין רשימות" in h for h in notes(app))
    app.text_input(key="pf_new_name").input("מעקב").run()
    app.button(key="pf_empty").click().run()
    assert not app.exception, app.exception
    assert app.session_state["pf_name"] == "מעקב"
    app.text_input(key="pf_add").input(f"{a.lower()}, {b} NOSUCH").run()
    app.button(key="pf_add_btn").click().run()
    assert not app.exception, app.exception
    saved = PF.load_all()["מעקב"]
    assert saved.kind == PF.WATCHLIST and [r["ticker"] for r in saved.rows] == [a, b]
    assert all(r["price"] is not None for r in saved.rows)                  # price when added
    assert any("NOSUCH" in toast.value for toast in app.toast)

    app.button(key="pf_show").click().run()
    assert not app.exception, app.exception
    assert app.session_state["page"] == "Screener" and app.session_state["query"] == f"{a}, {b}"
    assert any("2 תוצאות" in m.value for m in app.markdown)


def test_portfolio_values_and_trades(app):
    a, b, _ = open_portfolios(app)
    PF.save(PF.PList("תיק", PF.PORTFOLIO, [
        {"ticker": a, "date": "2025-01-02", "price": 10.0, "quantity": 10, "side": "buy", "fees": 0, "note": ""},
        {"ticker": "SPY", "date": "2025-01-02", "price": 500.0, "quantity": 1, "side": "buy", "fees": 0, "note": ""},
    ]))
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["pf_name"] == "תיק"
    price = float(app.session_state["df"].set_index("ticker").loc[a, "price"])
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["עלות"] == "$100"                                          # SPY has no price: left out
    assert metrics["שווי נוכחי"] == f"${price * 10:,.0f}"
    assert any("SPY" in h and "אין בסקרינר מחיר" in h for h in notes(app))

    # selling more than is held is refused, with a Hebrew message
    app.text_input(key="pf_t_ticker").input(a)
    app.number_input(key="pf_t_price").set_value(12.0)
    app.number_input(key="pf_t_qty").set_value(50.0)
    app.radio(key="pf_t_side").set_value("sell")
    app.button[[x.label for x in app.button].index("הוספת עסקה")].click().run()
    assert not app.exception, app.exception
    assert any("נמכרו יותר" in e.value for e in app.error)
    assert len(PF.load_all()["תיק"].rows) == 2

    app.text_input(key="pf_t_ticker").input(b)
    app.number_input(key="pf_t_price").set_value(20.0)
    app.number_input(key="pf_t_qty").set_value(5.0)
    app.radio(key="pf_t_side").set_value("buy")
    app.button[[x.label for x in app.button].index("הוספת עסקה")].click().run()
    assert not app.exception, app.exception
    rows = PF.load_all()["תיק"].rows
    assert rows[-1]["ticker"] == b and rows[-1]["quantity"] == 5 and rows[-1]["date"] == dt.date.today().isoformat()
    assert {m.label: m.value for m in app.metric}["עלות"] == "$200"

    app.button(key="pf_delete").click().run()
    assert not app.exception, app.exception
    assert PF.load_all() == {}


def test_import_problems_are_hebrew():
    from screener import i18n as I
    raw = pd.DataFrame({"סימול": ["AAPL", "AAPL", ""], "תאריך": ["2025-01-02", "2025-02-01", "2025-03-01"],
                        "שער": ["10", "abc", "5"], "כמות": ["1", "1", "1"]})
    texts = [I.problem(p) for p in PF.parse(raw).problems]
    assert texts == ["שורה 3: מספר לא תקין במחיר: abc", "שורה 4: חסר סימול"]
    assert I.problem(PF.Problem(0, "oversold", "AAPL")).startswith("ב-AAPL נמכרו")
