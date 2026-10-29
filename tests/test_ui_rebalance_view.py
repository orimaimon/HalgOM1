"""Tests for screener/ui/rebalance_view.py."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.ui.rebalance_view import calculate_rebalance, render_rebalance_view


def test_calculate_rebalance_balanced():
    # 2 positions, equal weights (50% each), target weights equal (50% each)
    df = pd.DataFrame([
        {"ticker": "AAPL", "value": 5000.0},
        {"ticker": "MSFT", "value": 5000.0},
    ])
    res = calculate_rebalance(df, threshold_pct=5.0)
    assert not res.empty
    assert len(res) == 2
    assert list(res["מניה"]) == ["AAPL", "MSFT"]
    assert list(res["משקל נוכחי %"]) == [50.0, 50.0]
    assert list(res["משקל יעד %"]) == [50.0, 50.0]
    assert list(res["סטייה %"]) == [0.0, 0.0]
    assert list(res["פעולה מומלצת"]) == ["ללא שינוי", "ללא שינוי"]
    assert list(res["שווי לשינוי $"]) == [0.0, 0.0]


def test_calculate_rebalance_drifted():
    # 2 positions, total value $10,000. AAPL = $8,000 (80%), MSFT = $2,000 (20%)
    # Target = 50% each -> drift AAPL = +30%, MSFT = -30%
    df = pd.DataFrame([
        {"ticker": "AAPL", "value": 8000.0},
        {"ticker": "MSFT", "value": 2000.0},
    ])
    res = calculate_rebalance(df, threshold_pct=5.0)
    assert not res.empty
    assert list(res["משקל נוכחי %"]) == [80.0, 20.0]
    assert list(res["משקל יעד %"]) == [50.0, 50.0]
    assert list(res["סטייה %"]) == [30.0, -30.0]
    assert list(res["פעולה מומלצת"]) == ["מכירה", "קנייה"]
    assert list(res["שווי לשינוי $"]) == [3000.0, 3000.0]


def test_calculate_rebalance_custom_targets():
    df = pd.DataFrame([
        {"ticker": "AAPL", "value": 7000.0},
        {"ticker": "MSFT", "value": 3000.0},
    ])
    targets = {"AAPL": 70.0, "MSFT": 30.0}
    res = calculate_rebalance(df, target_weights=targets, threshold_pct=5.0)
    assert list(res["פעולה מומלצת"]) == ["ללא שינוי", "ללא שינוי"]


def test_calculate_rebalance_empty():
    res = calculate_rebalance(pd.DataFrame())
    assert res.empty
    assert list(res.columns) == [
        "מניה",
        "משקל נוכחי %",
        "משקל יעד %",
        "סטייה %",
        "פעולה מומלצת",
        "שווי לשינוי $",
    ]


def test_ui_rebalance_view_balanced_render():
    df = pd.DataFrame([
        {"ticker": "AAPL", "value": 5000.0},
        {"ticker": "MSFT", "value": 5000.0},
    ])

    def script(pname, hdf):
        from screener.ui.rebalance_view import render_rebalance_view
        render_rebalance_view(pname, hdf)

    at = AppTest.from_function(script, args=("TestPort", df)).run()
    assert not at.exception

    # Subheader
    assert "איזון מחדש" in at.subheader[0].value
    # Success banner when all in range
    assert len(at.success) > 0
    assert "כל המניות בתיק נמצאות בטווח היעד" in at.success[0].value


def test_ui_rebalance_view_drifted_render():
    df = pd.DataFrame([
        {"ticker": "AAPL", "value": 8000.0},
        {"ticker": "MSFT", "value": 2000.0},
    ])

    def script(pname, hdf):
        from screener.ui.rebalance_view import render_rebalance_view
        render_rebalance_view(pname, hdf)

    at = AppTest.from_function(script, args=("TestPort", df)).run()
    assert not at.exception

    # Warning banner when drifted
    assert len(at.warning) > 0
    assert "נמצאו" in at.warning[0].value


def test_ui_rebalance_view_adjust_targets_validation():
    df = pd.DataFrame([
        {"ticker": "AAPL", "value": 5000.0},
        {"ticker": "MSFT", "value": 5000.0},
    ])

    saved_calls = []

    def on_save(name, targets):
        saved_calls.append((name, targets))

    def script(pname, hdf, cb):
        from screener.ui.rebalance_view import render_rebalance_view
        render_rebalance_view(pname, hdf, on_save_targets=cb)

    at = AppTest.from_function(script, args=("TestPort", df, on_save)).run()
    assert not at.exception

    # Set invalid target weights (e.g. 70% + 50% = 120%)
    at.number_input[1].set_value(70.0).run()  # AAPL target weight input
    assert len(at.error) > 0
    assert "סך משקלי היעד חייב להיות 100%" in at.error[0].value


def test_ui_rebalance_view_empty_holdings():
    def script():
        import pandas as pd
        from screener.ui.rebalance_view import render_rebalance_view
        render_rebalance_view("EmptyPort", pd.DataFrame())

    at = AppTest.from_function(script).run()
    assert not at.exception
    html_text = "".join(h.value for h in at.html)
    assert "אין פוזיציות פתוחות" in html_text


def test_ui_rebalance_view_error_handling():
    df = pd.DataFrame([{"ticker": "AAPL", "value": 5000.0}])

    def script(pname, hdf):
        import screener.ui.rebalance_view as rv
        orig = rv._render_rebalance_body
        try:
            rv._render_rebalance_body = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Rebalance failure"))
            rv.render_rebalance_view(pname, hdf)
        finally:
            rv._render_rebalance_body = orig

    at = AppTest.from_function(script, args=("ErrorPort", df)).run()
    assert not at.exception
    html_text = "".join(h.value for h in at.html)
    assert "שגיאה" in html_text
