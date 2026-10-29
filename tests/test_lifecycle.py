from __future__ import annotations

import logging
from unittest.mock import MagicMock

from screener import lifecycle


def test_is_server_mode(monkeypatch):
    monkeypatch.delenv("HALGOM_SERVER_MODE", raising=False)
    monkeypatch.delenv("HALGOM_NO_IDLE_SHUTDOWN", raising=False)
    assert not lifecycle.is_server_mode()

    monkeypatch.setenv("HALGOM_SERVER_MODE", "1")
    assert lifecycle.is_server_mode()

    monkeypatch.setenv("HALGOM_SERVER_MODE", "0")
    assert not lifecycle.is_server_mode()

    monkeypatch.setenv("HALGOM_NO_IDLE_SHUTDOWN", "1")
    assert lifecycle.is_server_mode()


def test_ensure_idle_watchdog_started_server_mode(monkeypatch, caplog):
    monkeypatch.setenv("HALGOM_SERVER_MODE", "1")
    monkeypatch.setattr(lifecycle, "_watchdog_thread", None)

    with caplog.at_level(logging.INFO):
        res = lifecycle.ensure_idle_watchdog_started()

    assert res is None
    assert "server mode: idle watchdog disabled" in caplog.text


def test_ensure_idle_watchdog_started_normal_mode(monkeypatch):
    monkeypatch.delenv("HALGOM_SERVER_MODE", raising=False)
    monkeypatch.delenv("HALGOM_NO_IDLE_SHUTDOWN", raising=False)

    # When thread is already alive, ensure_idle_watchdog_started returns existing thread
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    monkeypatch.setattr(lifecycle, "_watchdog_thread", mock_thread)

    res = lifecycle.ensure_idle_watchdog_started()
    assert res is mock_thread


def test_stop_server_server_mode(monkeypatch, caplog):
    monkeypatch.setenv("HALGOM_SERVER_MODE", "1")

    timer_mock = MagicMock()
    monkeypatch.setattr("threading.Timer", timer_mock)

    with caplog.at_level(logging.INFO):
        lifecycle.stop_server("unit test shutdown", delay=0.1)

    assert "server mode: stop_server ignored (unit test shutdown)" in caplog.text
    timer_mock.assert_not_called()


def test_stop_server_normal_mode(monkeypatch, caplog):
    monkeypatch.delenv("HALGOM_SERVER_MODE", raising=False)
    monkeypatch.delenv("HALGOM_NO_IDLE_SHUTDOWN", raising=False)

    timer_mock = MagicMock()
    monkeypatch.setattr("threading.Timer", timer_mock)

    with caplog.at_level(logging.INFO):
        lifecycle.stop_server("test normal shutdown", delay=0.1)

    assert "stopping server: test normal shutdown" in caplog.text
    timer_mock.assert_called_once()
def test_exit_button_hidden_in_server_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("HALGOM_SERVER_MODE", "1")
    monkeypatch.setenv("HALGOM_NO_AUTH", "1")

    def script():
        import datetime as dt
        from screener.data.store import Snapshot
        from screener.ui.common import _header
        snap = Snapshot(df=None, fetched_at=dt.datetime.now(), source="cache", meta={})
        _header(snap, 100)

    at = AppTest.from_function(script).run()
    assert not at.exception
    # Exit button should NOT be present in server mode
    assert len([b for b in at.button if b.key == "exit"]) == 0
