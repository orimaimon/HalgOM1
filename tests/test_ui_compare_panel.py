"""Tests for the compare panel UI component (screener/ui/compare_panel.py)."""
from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest


def _run_compare_panel(universe_dict: dict, candidates: list[str], selected: list[str] | None = None):
    """Function passed to AppTest.from_function."""
    import pandas as pd
    from screener.ui import compare_panel

    universe = pd.DataFrame(universe_dict)
    compare_panel.render(universe, candidates)


def test_compare_panel_two_tickers():
    universe_dict = {
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "price": [150.0, 300.0, None],
        "pe": [25.0, 30.0, 20.0],
        "market_cap": [2.5e12, 2.0e12, 1.5e12],
    }
    candidates = ["AAPL", "MSFT", "GOOGL"]

    def script(univ_dict, cands):
        import pandas as pd
        import streamlit as st
        from screener.ui import compare_panel

        universe = pd.DataFrame(univ_dict)
        # Pre-set selected tickers in session state or widget before render if needed,
        # or we can test multiselect interaction via AppTest
        compare_panel.render(universe, cands)

    at = AppTest.from_function(script, args=(universe_dict, candidates)).run()
    assert not at.exception

    # Select 2 tickers via multiselect
    ms = at.multiselect(key="cmp_tickers")
    assert ms is not None
    ms.select("AAPL").select("MSFT").run()
    assert not at.exception

    # Verify table rendered
    assert len(at.dataframe) == 1
    df_val = at.dataframe[0].value
    assert "AAPL" in df_val.columns
    assert "MSFT" in df_val.columns

    # Verify summary caption contains Hebrew
    assert len(at.caption) >= 1
    summary_text = at.caption[0].value
    assert "מובילה" in summary_text or "ב-" in summary_text


def test_compare_panel_one_ticker_shows_hint():
    universe_dict = {
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "price": [150.0, 300.0, 100.0],
    }
    candidates = ["AAPL", "MSFT"]

    def script(univ_dict, cands):
        import pandas as pd
        from screener.ui import compare_panel

        universe = pd.DataFrame(univ_dict)
        compare_panel.render(universe, cands)

    at = AppTest.from_function(script, args=(universe_dict, candidates)).run()
    assert not at.exception

    # Select 1 ticker
    at.multiselect(key="cmp_tickers").select("AAPL").run()
    assert not at.exception

    # Check hint is displayed and no dataframe
    assert len(at.dataframe) == 0
    # The note is rendered via st.html from hebrew_note
    html_texts = [h.value for h in at.html]
    assert any("בחרו 2 עד 4 מניות" in txt for txt in html_texts)


def test_compare_panel_compare_raises_error_handled():
    universe_dict = {
        "ticker": ["AAPL", "MSFT"],
        "price": [150.0, 300.0],
    }

    def script(univ_dict):
        import pandas as pd
        from screener import compare
        from screener.ui import compare_panel

        orig_compare = compare.compare
        try:
            def bad_compare(*args, **kwargs):
                raise RuntimeError("Simulated compare failure")

            compare.compare = bad_compare

            universe = pd.DataFrame(univ_dict)
            compare_panel.render(universe, ["AAPL", "MSFT"])
        finally:
            compare.compare = orig_compare

    at = AppTest.from_function(script, args=(universe_dict,)).run()
    at.multiselect(key="cmp_tickers").select("AAPL").select("MSFT").run()

    assert not at.exception
    assert len(at.dataframe) == 0
    html_texts = [h.value for h in at.html]
    assert any("שגיאה" in txt for txt in html_texts)


def test_render_stock_comparison_zero_tickers():
    universe_dict = {
        "ticker": ["AAPL", "MSFT", "NVDA", "AMZN"],
        "price": [150.0, 300.0, 120.0, 180.0],
    }

    def script(univ_dict):
        import pandas as pd
        from screener.ui import compare_panel

        universe = pd.DataFrame(univ_dict)
        compare_panel.render_stock_comparison(universe)

    at = AppTest.from_function(script, args=(universe_dict,)).run()
    assert not at.exception

    # 0 tickers selected by default
    assert len(at.dataframe) == 0
    assert len(at.metric) == 0
    html_texts = [h.value for h in at.html]
    assert any("בחרו 2 עד 5 מניות" in txt for txt in html_texts)


def test_render_stock_comparison_one_ticker():
    universe_dict = {
        "ticker": ["AAPL", "MSFT", "NVDA", "AMZN"],
        "price": [150.0, 300.0, 120.0, 180.0],
    }

    def script(univ_dict):
        import pandas as pd
        from screener.ui import compare_panel

        universe = pd.DataFrame(univ_dict)
        compare_panel.render_stock_comparison(universe, default_tickers=["AAPL"])

    at = AppTest.from_function(script, args=(universe_dict,)).run()
    assert not at.exception

    # 1 ticker selected
    assert len(at.dataframe) == 0
    assert len(at.metric) == 0
    html_texts = [h.value for h in at.html]
    assert any("בחרו 2 עד 5 מניות" in txt for txt in html_texts)


def test_render_stock_comparison_three_tickers():
    universe_dict = {
        "ticker": ["AAPL", "MSFT", "NVDA", "AMZN"],
        "price": [150.0, 300.0, 120.0, 180.0],
        "pe": [25.0, 30.0, 40.0, 50.0],
        "sales_growth": [0.10, 0.15, 0.80, 0.12],
        "score_value": [70.0, 60.0, 40.0, 30.0],
        "score_quality": [85.0, 90.0, 80.0, 75.0],
        "score_momentum": [60.0, 75.0, 95.0, 50.0],
    }

    def script(univ_dict):
        import pandas as pd
        from screener.ui import compare_panel

        universe = pd.DataFrame(univ_dict)
        compare_panel.render_stock_comparison(universe, default_tickers=["AAPL", "MSFT", "NVDA"])

    at = AppTest.from_function(script, args=(universe_dict,)).run()
    assert not at.exception

    # Check 4 metric cards rendered for dimension badges
    assert len(at.metric) == 4
    labels = [m.label for m in at.metric]
    assert "ערך" in labels
    assert "צמיחה" in labels
    assert "איכות" in labels
    assert "מומנטום" in labels

    # Check metric values (leaders)
    metrics_map = {m.label: m.value for m in at.metric}
    assert metrics_map["ערך"] == "AAPL"  # highest score_value
    assert metrics_map["צמיחה"] == "NVDA"  # highest sales_growth
    assert metrics_map["איכות"] == "MSFT"  # highest score_quality
    assert metrics_map["מומנטום"] == "NVDA"  # highest score_momentum

    # Check comparison dataframe rendered
    assert len(at.dataframe) == 1
    df_val = at.dataframe[0].value
    assert "AAPL" in df_val.columns
    assert "MSFT" in df_val.columns
    assert "NVDA" in df_val.columns
