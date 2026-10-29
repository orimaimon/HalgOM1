"""Tests for the Cash & Dividend Management UI component (screener/ui/cash_view.py)."""
from __future__ import annotations

import datetime as dt
import pytest
from streamlit.testing.v1 import AppTest


def script_render_empty(portfolio_name):
    from screener.ui.cash_view import render_cash_view
    render_cash_view(portfolio_name, [])


def script_render_normal(portfolio_name, entries):
    from screener.ui.cash_view import render_cash_view
    render_cash_view(portfolio_name, entries)


def script_render_form_submit(portfolio_name, added_container):
    from screener.ui.cash_view import render_cash_view

    def on_add(p_name, new_entry):
        added_container.append((p_name, new_entry))

    render_cash_view(portfolio_name, [], on_add_entry=on_add)


def script_render_error(portfolio_name):
    from screener.ui.cash_view import render_cash_view

    class BadEntry:
        @property
        def amount(self):
            raise RuntimeError("Corrupted entry field")

    render_cash_view(portfolio_name, [BadEntry()])


def test_render_cash_view_empty():
    at = AppTest.from_function(script_render_empty, args=("test_pf",)).run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Metrics check
    assert len(at.metric) == 3
    cash_metric = [m for m in at.metric if "יתרת מזומן" in m.label][0]
    dep_metric = [m for m in at.metric if "סך הפקדות" in m.label][0]
    div_metric = [m for m in at.metric if "דיבידנדים שהתקבלו" in m.label][0]

    assert cash_metric.value == "$0.00"
    assert dep_metric.value == "$0.00"
    assert div_metric.value == "$0.00"

    # Dataframe table check
    assert len(at.dataframe) == 1

    # Form check
    assert len(at.form) == 1


def test_render_cash_view_normal_entries():
    entries = [
        {"date": "2025-01-10", "kind": "הפקדה", "amount": 10000.0, "ticker": "", "note": "Initial deposit"},
        {"date": "2025-01-15", "kind": "משיכה", "amount": 2000.0, "ticker": "", "note": "Withdrawal"},
        {"date": "2025-02-01", "kind": "דיבידנד", "amount": 150.50, "ticker": "AAPL", "note": "Q1 dividend"},
        {"date": "2025-02-15", "kind": "ריבית", "amount": 25.00, "ticker": "", "note": "Interest credit"},
    ]

    at = AppTest.from_function(script_render_normal, args=("my_portfolio", entries)).run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Cash balance calculation: (10000 + 150.50 + 25) - 2000 = 8175.50
    cash_metric = [m for m in at.metric if "יתרת מזומן" in m.label][0]
    dep_metric = [m for m in at.metric if "סך הפקדות" in m.label][0]
    div_metric = [m for m in at.metric if "דיבידנדים שהתקבלו" in m.label][0]

    assert cash_metric.value == "$8,175.50"
    assert dep_metric.value == "$10,000.00"
    assert div_metric.value == "$150.50"

    # Dataframe check
    df_widget = at.dataframe[0]
    assert df_widget is not None


def test_render_cash_view_form_interaction():
    added = []
    at = AppTest.from_function(script_render_form_submit, args=("form_pf", added)).run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Fill form fields
    at.selectbox(key="cs_kind_form_pf").select("דיבידנד")
    at.number_input(key="cs_amount_form_pf").set_value(250.0)
    at.text_input(key="cs_ticker_form_pf").input("msft")
    at.text_input(key="cs_note_form_pf").input("Quarterly dividend")

    # Submit form
    at.form_submit_button(key="FormSubmitter:cs_form_form_pf-הוספת תנועה").click().run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Verify on_add callback was invoked
    assert len(added) == 1
    pf_name, new_entry = added[0]
    assert pf_name == "form_pf"
    assert new_entry["kind"] == "דיבידנד"
    assert new_entry["amount"] == 250.0
    assert new_entry["ticker"] == "MSFT"
    assert new_entry["note"] == "Quarterly dividend"


def test_render_cash_view_exception_handling():
    at = AppTest.from_function(script_render_error, args=("error_pf",)).run()

    # Exception should be handled internally by render_cash_view without crashing
    assert not at.exception
