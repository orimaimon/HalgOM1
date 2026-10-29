"""
Tests for screener/ui/rs_rating_panel.py component.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _make_df() -> pd.DataFrame:
    rows = [
        {"ticker": "AAPL", "rs_rating": 95, "price": 180.0, "sector": "Technology Services"},
        {"ticker": "MSFT", "rs_rating": 88, "price": 400.0, "sector": "Technology Services"},
        {"ticker": "NVDA", "rs_rating": 99, "price": 120.0, "sector": "Electronic Technology"},
        {"ticker": "JPM", "rs_rating": 45, "price": 190.0, "sector": "Finance"},
        {"ticker": "XOM", "rs_rating": 20, "price": 110.0, "sector": "Energy Minerals"},
        {"ticker": "TSLA", "rs_rating": 10, "price": 170.0, "sector": "Consumer Durables"},
    ]
    return pd.DataFrame(rows)


def _script_render(df=None, top_n=15, monkeypatch_error=False):
    import streamlit as st
    from screener.ui import rs_rating_panel

    if monkeypatch_error:
        def _raise(*args, **kwargs):
            raise RuntimeError("simulated error")
        import pandas as pd
        monkeypatch_error_fn = _raise
        # Force error inside render
        old_sort = pd.DataFrame.sort_values
        pd.DataFrame.sort_values = monkeypatch_error_fn
        try:
            rs_rating_panel.render(df, top_n=top_n)
        finally:
            pd.DataFrame.sort_values = old_sort
        return

    rs_rating_panel.render(df, top_n=top_n)


def test_rs_rating_panel_normal_frame():
    df = _make_df()
    at = AppTest.from_function(_script_render, args=(df, 15)).run()
    assert not at.exception

    # Headers and tables
    assert any("דירוג עוצמה יחסית (RS Rating)" in sub.value for sub in at.subheader)
    assert len(at.dataframe) == 2
    charts = at.get("plotly_chart")
    assert len(charts) >= 1 or any(c.key == "rs_histogram_chart" for c in charts)


def test_rs_rating_panel_missing_column():
    df = _make_df().drop(columns=["rs_rating"])
    at = AppTest.from_function(_script_render, args=(df, 15)).run()
    assert not at.exception

    # Info note displayed, tables not rendered
    html_texts = [h.value for h in at.html]
    assert any("אין נתוני דירוג עוצמה יחסית (RS Rating) להצגה." in text for text in html_texts)
    assert len(at.dataframe) == 0


def test_rs_rating_panel_empty_frame():
    df = pd.DataFrame(columns=["ticker", "rs_rating", "price", "sector"])
    at = AppTest.from_function(_script_render, args=(df, 15)).run()
    assert not at.exception

    html_texts = [h.value for h in at.html]
    assert any("אין נתוני דירוג עוצמה יחסית (RS Rating) להצגה." in text for text in html_texts)
    assert len(at.dataframe) == 0


def test_rs_rating_panel_none_df():
    at = AppTest.from_function(_script_render, args=(None, 15)).run()
    assert not at.exception

    html_texts = [h.value for h in at.html]
    assert any("אין נתוני דירוג עוצמה יחסית (RS Rating) להצגה." in text for text in html_texts)
    assert len(at.dataframe) == 0


def test_rs_rating_panel_all_nan():
    df = _make_df()
    df["rs_rating"] = np.nan
    at = AppTest.from_function(_script_render, args=(df, 15)).run()
    assert not at.exception

    html_texts = [h.value for h in at.html]
    assert any("אין נתוני דירוג עוצמה יחסית (RS Rating) להצגה." in text for text in html_texts)
    assert len(at.dataframe) == 0


def test_rs_rating_panel_top_n_larger_than_frame():
    df = _make_df()  # 6 rows
    at = AppTest.from_function(_script_render, args=(df, 15)).run()
    assert not at.exception

    assert len(at.dataframe) == 2


def test_rs_rating_panel_exception_handling():
    df = _make_df()
    at = AppTest.from_function(_script_render, args=(df, 15, True)).run()
    assert not at.exception

    html_texts = [h.value for h in at.html]
    assert any("שגיאה בהצגת דירוג עוצמה יחסית" in text for text in html_texts)
