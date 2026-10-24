"""Tests for screener/ui/explain_panel.py."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener import i18n
from screener.derive import add_derived
from tests.test_screener import TODAY, make_raw


def make_test_row(**kwargs) -> pd.Series:
    raw = make_raw(n=1, seed=42)
    df = add_derived(raw, today=TODAY)
    row = df.iloc[0].copy()
    for k, v in kwargs.items():
        row[k] = v
    return row


def test_ui_explain_panel_active_selections_pass_and_fail():
    row = make_test_row(
        price=150.0,
        pe=12.0,
        market_cap=2.1e9,
        change=12.3,
    )
    selections = {"pe": "Under 15", "price": "Under $50"}

    def script(r, sel):
        from screener.ui.explain_panel import render
        render(r, sel)

    at = AppTest.from_function(script, args=(row, selections)).run()
    assert not at.exception
    # Check description caption
    assert len(at.caption) >= 1
    assert any("חברה" in c.value for c in at.caption)

    # Check HTML rendered lines inside expander
    assert len(at.html) >= 1
    html_content = "".join(h.value for h in at.html)
    assert "✓" in html_content
    assert "✗" in html_content
    assert "$150.00" in html_content


def test_ui_explain_panel_no_active_filters():
    row = make_test_row()
    selections = {}

    def script(r, sel):
        from screener.ui.explain_panel import render
        render(r, sel)

    at = AppTest.from_function(script, args=(row, selections)).run()
    assert not at.exception
    assert any("אין פילטרים פעילים" in m.value for m in at.markdown)


def test_ui_explain_panel_escapes_html_in_data():
    row = make_test_row(company="<b>EvilCorp</b>", sector="<b>Tech</b>", pe=10.0)
    selections = {"sector": "<b>Tech</b>"}

    def script(r, sel):
        from screener.ui.explain_panel import render
        render(r, sel)

    at = AppTest.from_function(script, args=(row, selections)).run()
    assert not at.exception
    html_content = "".join(h.value for h in at.html)
    # The raw unescaped <b>Tech</b> should not be in the output; it should be escaped as &lt;b&gt;Tech&lt;/b&gt;
    assert "<b>Tech</b>" not in html_content
    assert "&lt;b&gt;Tech&lt;/b&gt;" in html_content


def test_ui_explain_panel_explain_raising_shows_error_note():
    row = make_test_row()
    selections = {"pe": "Under 15"}

    def script(r, sel):
        import screener.explain as ex
        orig = ex.explain
        try:
            ex.explain = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("explain failed"))
            from screener.ui.explain_panel import render
            render(r, sel)
        finally:
            ex.explain = orig

    at = AppTest.from_function(script, args=(row, selections)).run()
    assert not at.exception
    assert any("שגיאה" in h.value for h in at.html)


def test_ui_explain_panel_nan_value_shows_dash():
    row = make_test_row(pe=np.nan)
    selections = {"pe": "Under 15"}

    def script(r, sel):
        from screener.ui.explain_panel import render
        render(r, sel)

    at = AppTest.from_function(script, args=(row, selections)).run()
    assert not at.exception
    html_content = "".join(h.value for h in at.html)
    assert "—" in html_content


def test_ui_explain_panel_english_language():
    row = make_test_row(price=10.0, pe=12.0)
    selections = {"pe": "Under 15"}

    def script(r, sel):
        import streamlit as st
        st.session_state["lang"] = "en"
        from screener.ui.explain_panel import render
        render(r, sel)

    at = AppTest.from_function(script, args=(row, selections)).run()
    assert not at.exception
    html_content = "".join(h.value for h in at.html)
    assert "P/E" in html_content or "12" in html_content
