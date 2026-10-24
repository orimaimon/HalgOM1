"""Tests for the Alerts manager UI component (screener/ui/alerts_view.py)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener import alerts
from screener.ui import alerts_view


def script_render_alerts(universe, session_day, alerts_path):
    import datetime as dt
    from pathlib import Path
    from screener import alerts
    from screener.ui.alerts_view import render

    alerts.ALERTS_PATH = Path(alerts_path)
    render(universe, session_day)


def script_render_error(universe, session_day):
    from screener import alerts
    from screener.ui.alerts_view import render

    orig_load = alerts.load
    def bad_load(*args, **kwargs):
        raise RuntimeError("Storage failure")

    alerts.load = bad_load
    try:
        render(universe, session_day)
    finally:
        alerts.load = orig_load


def test_fired_today_and_deduplication(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    # Save an alert that will fire
    a = alerts.Alert(
        id="a1",
        ticker="AAPL",
        kind="price_below",
        value=150.0,
        note="buy dip",
    )
    alerts.add(a, alerts_file)

    universe = pd.DataFrame([{"ticker": "AAPL", "price": 100.0, "change": -2.0}])
    session_day = dt.date(2025, 3, 14)

    # First run on session_day: should fire, show info message, and mark last_fired
    at = AppTest.from_function(script_render_alerts, args=(universe, session_day, str(alerts_file)))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception
    assert len(at.info) == 1
    assert "AAPL" in at.info[0].value
    assert "100.00" in at.info[0].value

    loaded = alerts.load(alerts_file)
    assert loaded["a1"].last_fired == "2025-03-14"

    # Second run on the same session_day: no info message should be shown
    at2 = AppTest.from_function(script_render_alerts, args=(universe, session_day, str(alerts_file)))
    at2.run()
    if at2.exception:
        raise at2.exception[0]
    assert not at2.exception
    assert len(at2.info) == 0


def test_add_form_valid_and_invalid_ticker(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    universe = pd.DataFrame([{"ticker": "AAPL", "price": 100.0}])
    session_day = dt.date(2025, 3, 14)

    # Test unknown ticker
    at = AppTest.from_function(script_render_alerts, args=(universe, session_day, str(alerts_file)))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    at.text_input(key="al_ticker").input("UNKNOWN")
    at.button(key="al_submit").click().run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception
    assert len(at.error) == 1
    assert "UNKNOWN" in at.error[0].value
    assert len(alerts.load(alerts_file)) == 0

    # Test valid ticker addition
    at2 = AppTest.from_function(script_render_alerts, args=(universe, session_day, str(alerts_file)))
    at2.run()
    at2.text_input(key="al_ticker").input("AAPL")
    at2.selectbox(key="al_kind").select("price_below")
    at2.number_input(key="al_value").set_value(95.0)
    at2.text_input(key="al_note").input("test note")
    at2.button(key="al_submit").click().run()
    if at2.exception:
        raise at2.exception[0]
    assert not at2.exception

    loaded = alerts.load(alerts_file)
    assert len(loaded) == 1
    new_a = list(loaded.values())[0]
    assert new_a.ticker == "AAPL"
    assert new_a.kind == "price_below"
    assert new_a.value == 95.0
    assert new_a.note == "test note"


def test_remove_alert(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    a = alerts.Alert(id="a_rm_test", ticker="MSFT", kind="price_above", value=400.0)
    alerts.add(a, alerts_file)

    universe = pd.DataFrame([{"ticker": "MSFT", "price": 350.0}])
    session_day = dt.date(2025, 3, 14)

    at = AppTest.from_function(script_render_alerts, args=(universe, session_day, str(alerts_file)))
    at.run()
    if at.exception:
        raise at.exception[0]

    # Click remove button for a_rm_test
    at.button(key="al_rm_a_rm_test").click().run()
    if at.exception:
        raise at.exception[0]

    loaded = alerts.load(alerts_file)
    assert len(loaded) == 0


def test_no_alerts_and_error_path(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    universe = pd.DataFrame([{"ticker": "AAPL", "price": 100.0}])
    session_day = dt.date(2025, 3, 14)

    # 1. No alerts -> "אין התראות עדיין" note
    at = AppTest.from_function(script_render_alerts, args=(universe, session_day, str(alerts_file)))
    at.run()
    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # 2. Storage exception -> Hebrew note, no top-level exception
    at_err = AppTest.from_function(script_render_error, args=(universe, session_day))
    at_err.run()
    assert not at_err.exception


def test_fired_today_pure_function(tmp_path, monkeypatch):
    alerts_file = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "ALERTS_PATH", alerts_file)

    a = alerts.Alert(id="a1", ticker="NVDA", kind="change_above", value=5.0)
    alerts.add(a, alerts_file)

    universe = pd.DataFrame([{"ticker": "NVDA", "change": 7.5}])
    session_day = dt.date(2025, 3, 14)

    fired = alerts_view.fired_today(universe, session_day)
    assert len(fired) == 1
    assert fired[0].ticker == "NVDA"

    # Verify fired_today did not mark last_fired in storage
    loaded = alerts.load(alerts_file)
    assert loaded["a1"].last_fired is None
