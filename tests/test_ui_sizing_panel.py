"""Tests for screener/ui/sizing_panel.py."""
import math
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener.ui.sizing_panel import calculate_sizing, render


def test_calculate_sizing_standard_inputs():
    res = calculate_sizing(price=100.0, atr=2.5, account=50000.0, risk_pct=1.0, atr_mult=2.0, target_r=2.0)
    assert res is not None
    assert res["stop_dist"] == 5.0
    assert res["stop_price"] == 95.0
    assert res["stop_pct"] == -5.0
    assert res["risk_dollars"] == 500.0
    assert res["shares"] == 100
    assert res["cost"] == 10000.0
    assert res["weight_pct"] == 20.0
    assert res["actual_risk"] == 500.0
    assert res["target_price"] == 110.0
    assert res["target_pct"] == 10.0


def test_calculate_sizing_capped_by_account_size():
    # Large risk dollar allows 500 shares, but account ($1000) only allows 10 shares at $100 price
    res = calculate_sizing(price=100.0, atr=0.1, account=1000.0, risk_pct=5.0, atr_mult=1.0)
    assert res is not None
    assert res["shares"] == 10
    assert res["cost"] == 1000.0
    assert res["weight_pct"] == 100.0


def test_calculate_sizing_invalid_inputs():
    # None inputs
    assert calculate_sizing(price=None, atr=2.5, account=50000.0) is None
    assert calculate_sizing(price=100.0, atr=None, account=50000.0) is None
    assert calculate_sizing(price=100.0, atr=2.5, account=None) is None

    # NaN inputs
    assert calculate_sizing(price=np.nan, atr=2.5, account=50000.0) is None
    assert calculate_sizing(price=100.0, atr=np.nan, account=50000.0) is None

    # <= 0 inputs
    assert calculate_sizing(price=0.0, atr=2.5, account=50000.0) is None
    assert calculate_sizing(price=-10.0, atr=2.5, account=50000.0) is None
    assert calculate_sizing(price=100.0, atr=0.0, account=50000.0) is None
    assert calculate_sizing(price=100.0, atr=2.5, account=0.0) is None
    assert calculate_sizing(price=100.0, atr=2.5, account=50000.0, risk_pct=0.0) is None
    assert calculate_sizing(price=100.0, atr=2.5, account=50000.0, atr_mult=0.0) is None

    # ATR stop distance >= price
    assert calculate_sizing(price=10.0, atr=6.0, account=50000.0, atr_mult=2.0) is None


def test_ui_sizing_panel_render_valid_row():
    row = pd.Series({"ticker": "NVDA", "price": 100.0, "atr": 2.5})

    def script(r):
        from screener.ui.sizing_panel import render
        render(r)

    at = AppTest.from_function(script, args=(row,)).run()
    assert not at.exception

    # Should render 3 number inputs
    assert len(at.number_input) == 3
    assert at.number_input[0].key == "sz_account"
    assert at.number_input[1].key == "sz_risk_pct"
    assert at.number_input[2].key == "sz_atr_mult"

    # Should render 4 metrics
    assert len(at.metric) == 4
    # 100 shares
    assert at.metric[0].value == "100"
    # $95.00 stop
    assert at.metric[1].value == "$95.00"
    # $110.00 target
    assert at.metric[2].value == "$110.00"
    # $500 risk
    assert at.metric[3].value == "$500"


def test_ui_sizing_panel_update_risk_pct():
    row = pd.Series({"ticker": "NVDA", "price": 100.0, "atr": 2.5})

    def script(r):
        from screener.ui.sizing_panel import render
        render(r)

    at = AppTest.from_function(script, args=(row,)).run()
    assert not at.exception

    # Change risk_pct to 2.0
    at.number_input[1].set_value(2.0).run()
    assert not at.exception

    # Shares updated to 200
    assert at.metric[0].value == "200"


def test_ui_sizing_panel_missing_atr():
    row = pd.Series({"ticker": "NVDA", "price": 100.0, "atr": np.nan})

    def script(r):
        from screener.ui.sizing_panel import render
        render(r)

    at = AppTest.from_function(script, args=(row,)).run()
    assert not at.exception

    # No metrics rendered
    assert len(at.metric) == 0
    # Hebrew note shown
    html_content = "".join(h.value for h in at.html)
    assert "אין נתוני ATR" in html_content


def test_ui_sizing_panel_error_handling():
    row = pd.Series({"ticker": "NVDA", "price": 100.0, "atr": 2.5})

    def script(r):
        import screener.ui.sizing_panel as sp
        orig = sp.calculate_sizing
        try:
            sp.calculate_sizing = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Calculation failed"))
            sp.render(r)
        finally:
            sp.calculate_sizing = orig

    at = AppTest.from_function(script, args=(row,)).run()
    assert not at.exception
    html_content = "".join(h.value for h in at.html)
    assert "שגיאה" in html_content
