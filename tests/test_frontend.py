"""Front-end behaviour that needs the running app (Streamlit AppTest, offline, synthetic data)."""
from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from screener import export, lifecycle
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
