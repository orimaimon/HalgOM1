"""Tests for screener/ui/tax_view.py component using Streamlit AppTest.
"""
from __future__ import annotations

import datetime as dt
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def test_tax_view_empty_data():
    tax_df = pd.DataFrame()
    summary = {}
    year = 2025

    def script(df, s, y):
        from screener.ui.tax_view import render_tax_report
        render_tax_report(df, s, y)

    at = AppTest.from_function(script, args=(tax_df, summary, year)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Year selector
    sel = at.selectbox(key="tx_year")
    assert sel.value == 2025

    # 4 metric cards present
    metrics = at.metric
    assert len(metrics) >= 4

    m_proceeds = [m for m in metrics if "סך תמורה" in m.label][0]
    m_gain = [m for m in metrics if "רווח הון" in m.label][0]
    m_loss = [m for m in metrics if "הפסדים לקיזוז" in m.label][0]
    m_tax = [m for m in metrics if "חבות מס משוערת" in m.label][0]

    assert m_proceeds.value == "₪0.00"
    assert m_gain.value == "₪0.00"
    assert m_loss.value == "₪0.00"
    assert m_tax.value == "₪0.00"

    # Closed trades table present
    tables = at.get("dataframe")
    assert len(tables) > 0

    # Legal disclaimer present
    info_boxes = at.info
    assert any("החישוב נועד לאומדן כללי בלבד" in box.value for box in info_boxes)


def test_tax_view_positive_gains():
    tax_df = pd.DataFrame([
        {
            "ticker": "AAPL",
            "buy_date": dt.date(2024, 5, 10),
            "buy_rate": 3.70,
            "sell_date": dt.date(2025, 2, 15),
            "sell_rate": 3.65,
            "nominal_gain_ils": 5000.0,
            "estimated_tax_ils": 1250.0,
            "proceeds_ils": 20000.0,
        },
        {
            "ticker": "NVDA",
            "buy_date": "2024-06-01",
            "buy_rate": 3.72,
            "sell_date": "2025-03-20",
            "sell_rate": 3.68,
            "nominal_gain_ils": 3000.0,
            "estimated_tax_ils": 750.0,
            "proceeds_ils": 15000.0,
        },
    ])

    summary = {
        "total_proceeds_ils": 35000.0,
        "net_gain_ils": 8000.0,
        "total_loss_ils": 0.0,
        "estimated_tax_ils": 2000.0,
    }
    year = 2025

    def script(df, s, y):
        from screener.ui.tax_view import render_tax_report
        render_tax_report(df, s, y)

    at = AppTest.from_function(script, args=(tax_df, summary, year)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    metrics = at.metric
    m_proceeds = [m for m in metrics if "סך תמורה" in m.label][0]
    m_gain = [m for m in metrics if "רווח הון" in m.label][0]
    m_loss = [m for m in metrics if "הפסדים לקיזוז" in m.label][0]
    m_tax = [m for m in metrics if "חבות מס משוערת" in m.label][0]

    assert m_proceeds.value == "₪35,000.00"
    assert m_gain.value == "₪8,000.00"
    assert m_loss.value == "₪0.00"
    assert m_tax.value == "₪2,000.00"

    info_boxes = at.info
    assert any("החישוב נועד לאומדן כללי בלבד" in box.value for box in info_boxes)


def test_tax_view_net_loss_year():
    tax_df = pd.DataFrame([
        {
            "ticker": "TSLA",
            "buy_date": dt.date(2025, 1, 10),
            "buy_rate": 3.65,
            "sell_date": dt.date(2025, 4, 12),
            "sell_rate": 3.60,
            "nominal_gain_ils": -4000.0,
            "estimated_tax_ils": 0.0,
            "proceeds_ils": 10000.0,
        },
    ])

    summary = {
        "total_proceeds_ils": 10000.0,
        "total_gain_ils": 0.0,
        "total_loss_ils": 4000.0,
        "net_gain_ils": -4000.0,
        "estimated_tax_ils": 0.0,
    }
    year = 2025

    def script(df, s, y):
        from screener.ui.tax_view import render_tax_report
        render_tax_report(df, s, y)

    at = AppTest.from_function(script, args=(tax_df, summary, year)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    metrics = at.metric
    m_loss = [m for m in metrics if "הפסדים לקיזוז" in m.label][0]
    m_tax = [m for m in metrics if "חבות מס משוערת" in m.label][0]

    assert m_loss.value == "₪4,000.00"
    assert m_tax.value == "₪0.00"


def test_tax_view_error_handling():
    tax_df = pd.DataFrame()
    summary = {}
    year = 2025

    def script(df, s, y):
        from screener.ui import tax_view
        def broken_columns(*args, **kwargs):
            raise RuntimeError("Simulated error in UI")

        st_old = tax_view.st
        try:
            tax_view.st = None  # cause exception
            tax_view.render_tax_report(df, s, y)
        finally:
            tax_view.st = st_old

    at = AppTest.from_function(script, args=(tax_df, summary, year)).run()

    assert not at.exception
