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
        "signals": 300, "indexes": 140, "new": 60
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
    assert not any("zoom: 1.15" in h for h in notes(app))

    # Enable Large Text
    app.toggle(key="large_text").set_value(True).run()
    assert app.session_state["large_text"]
    assert any("zoom: 1.15" in h for h in notes(app))


def test_data_time_header(app):
    app.run()
    assert not app.exception, app.exception
    assert any("שעת הנתונים:" in c.value for c in app.caption)


def test_views_price_change_order():
    from screener.fields import VIEWS

    for view, cols in VIEWS.items():
        start = ["ticker", "company"] if "company" in cols else ["ticker"]
        expected_prefix = start + ["price", "change"]
        assert cols[:len(expected_prefix)] == expected_prefix, f"View {view} order is {cols[:4]}"


def test_escape_markdown_and_clean_str():
    from screener.app import _clean_str, escape_markdown

    assert escape_markdown("Apple $Inc.* _Co_") == "Apple \\$Inc.\\* \\_Co\\_"
    assert escape_markdown(None) == "—"
    assert escape_markdown("nan") == "—"

    assert _clean_str(float("nan")) == "—"
    assert _clean_str(None) == "—"
    assert _clean_str("nan") == "—"
    assert _clean_str("Technology") == "Technology"


def test_filter_panel_collapsed_by_default(app):
    app.run()
    assert not app.exception, app.exception
    # Expander for filters should be collapsed when 0 filters are active
    exp = app.expander[0]
    assert exp.proto.expanded is False

    # Activating a filter expands it
    app.selectbox(key="f_pe").select("Low (<15)").run()
    assert app.expander[0].proto.expanded is True


def test_save_bookmark_button_label(app):
    app.run()
    assert not app.exception, app.exception
    # Popover button for saving screen should have label "שמירה"
    assert any("שמירה" in b.label for b in app.button)


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


# ── round 3, part B1 (#43) ──────────────────────────────────────────────────

def test_one_ticker_with_a_comma_is_an_exact_search():
    from screener.filters import search
    df = pd.DataFrame({"ticker": ["F", "PFE", "AFL"], "company": ["Ford", "Pfizer", "Aflac"]})
    assert list(search(df, "F,")["ticker"]) == ["F"]
    assert list(search(df, "F")["ticker"]) == ["F", "PFE", "AFL"]          # a word is still a substring search


def test_show_one_ticker_list_in_screener(app):
    a, _, _ = open_portfolios(app)
    PF.save(PF.PList("אחת", PF.WATCHLIST, [{"ticker": a, "date": "2026-10-01", "price": 10.0, "note": ""}]))
    app.session_state["only_new"] = True
    app.run()
    app.button(key="pf_show").click().run()
    assert not app.exception, app.exception
    assert app.session_state["hide_junk"] is False and app.session_state["only_new"] is False
    assert any("1 תוצאות" in m.value for m in app.markdown)
    assert any("מוצגת הרשימה" in h for h in notes(app))
    app.button(key="list_banner_close").click().run()
    assert app.session_state["query"] == "" and not any("מוצגת הרשימה" in h for h in notes(app))


def test_reset_turns_only_new_off(app):
    app.run()
    app.session_state["only_new"] = True
    app.run()
    next(b for b in app.button if b.label == "איפוס").click().run()
    assert not app.exception, app.exception
    assert app.session_state["only_new"] is False


def test_changed_preset_is_marked_and_can_be_reapplied(app):
    app.run()
    app.selectbox(key="preset").select("Unusual volume").run()
    assert "preset_reapply" not in [b.key for b in app.button]
    app.selectbox(key="f_rel_volume").select("Any").run()                  # change the preset's filter
    assert not app.exception, app.exception
    assert any("(שונה)" in o for o in app.selectbox(key="preset").options)   # options are the shown labels
    app.button(key="preset_reapply").click().run()
    assert app.session_state["f_rel_volume"] != "Any"
    assert "preset_reapply" not in [b.key for b in app.button]


def test_results_count_follows_only_new(app, tmp_path):
    import datetime as dt

    import screener.data.store as store_mod
    raw = clean(make_raw(300).assign(is_primary=True, currency="USD"))
    snaps = tmp_path / "snapshots"
    snaps.mkdir(exist_ok=True)
    raw.head(250).to_parquet(snaps / f"{store_mod.session_date(dt.datetime.now() - dt.timedelta(days=4))}.parquet")
    app.run()
    app.toggle(key="only_new").set_value(True).run()
    assert not app.exception, app.exception
    shown = len(app.dataframe[0].value) if app.dataframe else 0
    assert any(f"{shown} תוצאות" in m.value for m in app.markdown)


def test_goodbye_mentions_other_tabs(app, monkeypatch):
    monkeypatch.setattr(lifecycle, "stop_server", lambda reason: None)
    app.run()
    app.button(key="exit").click().run()
    assert any("לשוניות אחרות" in h for h in notes(app))


# ── round 3, part B2 ────────────────────────────────────────────────────────

def test_regime_reasons_translation():
    from screener import i18n as I
    from screener import market as M

    m = pd.DataFrame({"close": [500.0, 400, 80, 15], "sma20": [490.0, 390, 79, 16],
                       "sma200": [450.0, 350, 78, 17]}, index=["SPY", "QQQ", "HYG", "VIX"])
    regime = M.compute_regime(m)
    # Internal reasons remain English
    assert regime.reasons == ["VIX 15.0 (<18 calm)", "SPY above SMA200", "HYG above SMA20 (credit risk-on)"]
    # Translated reasons for display
    translated = [I.regime_reason(r, I.HE) for r in regime.reasons]
    assert translated == ["VIX 15.0 (<18 רגוע)", "SPY מעל SMA200", "HYG מעל SMA20 (אשראי risk-on)"]


def test_chart_locale(app, monkeypatch):
    from screener.app import tradingview_chart
    iframe_args = []
    monkeypatch.setattr(st, "iframe", lambda src, height=None: iframe_args.append(src))

    app.run()
    tradingview_chart("AAPL")
    assert len(iframe_args) == 1
    assert '"locale": "he_IL"' in iframe_args[0]


def test_save_screen_overwrite_warning(app):
    app.run()
    # Save a user preset for the first time
    P.save_user("MySavedScreen", P.Preset(filters={}, view="Overview", sort="ticker", ascending=True))
    app.run()

    # Enter existing name and click save
    app.text_input(key="save_name").input("MySavedScreen").run()
    app.button[[b.label for b in app.button].index("שמירה")].click().run()
    assert not app.exception, app.exception

    # First click sets confirmation state and displays warning
    assert app.session_state.get("confirm_save_name") == "MySavedScreen"
    assert any("סריקה בשם זה כבר קיימת" in w.value for w in app.warning)

    # Second click completes overwrite
    app.button[[b.label for b in app.button].index("שמירה")].click().run()
    assert not app.exception, app.exception
    assert app.session_state.get("confirm_save_name") is None
    assert app.session_state["preset"] == "★ MySavedScreen"


def test_junk_conflict_hint(app):
    app.run()
    assert not any("מתנגשת עם “הסתר זבל”" in h for h in notes(app))

    # Select Nano (under $50M) when Hide junk is on
    app.selectbox(key="f_market_cap").select("Nano (under $50M)").run()
    assert not app.exception, app.exception
    assert any("מתנגשת עם “הסתר זבל”" in h for h in notes(app))

    # Turn off Hide junk: conflict hint disappears
    app.toggle(key="hide_junk").set_value(False).run()
    assert not app.exception, app.exception
    assert not any("מתנגשת עם “הסתר זבל”" in h for h in notes(app))


def test_pipeline_cached_on_rerun(app):
    app.run()
    assert "pipeline_key" in app.session_state
    assert "pipeline_res" in app.session_state
    res_first = app.session_state["pipeline_res"]

    # Rerun without changing filters
    app.run()
    assert app.session_state["pipeline_res"] is res_first


def test_search_as_removable_pill(app):
    app.run()
    app.text_input(key="query").input("AAPL").run()
    assert not app.exception, app.exception

    pills = next(p for p in app.get("button_group") if (p.key or "").startswith("active_pills_"))
    assert any("חיפוש: AAPL" in o for o in pills.options)

    # Deselecting search pill clears search query
    pills.set_value([]).run()
    assert not app.exception, app.exception
    assert app.session_state["query"] == ""


def test_regime_reasons_translation_below():
    from screener import i18n as I
    assert I.regime_reason("SPY below SMA200", I.HE) == "SPY מתחת ל-SMA200"
    assert I.regime_reason("HYG below SMA20 (credit risk-off)", I.HE) == "HYG מתחת ל-SMA20 (אשראי risk-off)"
    assert I.regime_reason("SPY below SMA200", I.EN) == "SPY below SMA200"
