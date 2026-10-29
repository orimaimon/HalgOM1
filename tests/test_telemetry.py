"""
Unit tests for screener.telemetry module.
"""
from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from screener import telemetry, whatsnew


def test_sanitize_traceback_home_dir():
    # Test path replacement for home directory
    raw_tb = (
        'File "/home/david/project/screener/app.py", line 42, in main\n'
        'File "C:\\Users\\david_haim\\screener\\app.py", line 10, in run\n'
        "  raise ValueError('crash')"
    )
    sanitized = telemetry.sanitize_traceback(raw_tb)
    assert "/home/david" not in sanitized
    assert "C:\\Users\\david_haim" not in sanitized
    assert "~/project/screener/app.py" in sanitized
    assert "~\\screener\\app.py" in sanitized


def test_sanitize_traceback_sensitive_tokens():
    raw_tb = (
        "ValueError: invalid request with api_key='secret_key_12345' "
        "and token: 'abc-xyz-token' "
        "Authorization: Bearer secret_bearer_token_999"
    )
    sanitized = telemetry.sanitize_traceback(raw_tb)
    assert "secret_key_12345" not in sanitized
    assert "abc-xyz-token" not in sanitized
    assert "secret_bearer_token_999" not in sanitized
    assert "[MASKED]" in sanitized


def test_sanitize_traceback_empty():
    assert telemetry.sanitize_traceback("") == ""
    assert telemetry.sanitize_traceback(None) == ""


def test_build_crash_report_fields():
    try:
        raise ValueError("Something went wrong in portfolio calculations")
    except ValueError as e:
        exc = e

    report = telemetry.build_crash_report(exc)

    assert report["exception_type"] == "ValueError"
    assert report["exception_message"] == "Something went wrong in portfolio calculations"
    assert report["app_version"] == whatsnew.version_label()
    assert report["python_version"] == sys.version
    assert isinstance(report["os_info"], str)
    assert isinstance(report["timestamp"], str)
    assert "ValueError: Something went wrong" in report["traceback"]
    assert "extra_context" not in report


def test_build_crash_report_extra_context():
    try:
        raise RuntimeError("Database error")
    except RuntimeError as e:
        exc = e

    ctx = {"user_action": "click_button", "selected_ticker": "AAPL"}
    report = telemetry.build_crash_report(exc, extra_context=ctx)

    assert report["extra_context"] == ctx


def test_send_crash_webhook_no_url(monkeypatch):
    monkeypatch.delenv("HALGOM_WEBHOOK_URL", raising=False)
    report = {"test": "data"}
    res = telemetry.send_crash_webhook(report, webhook_url=None)
    assert res is False


def test_send_crash_webhook_success():
    report = {"exception_type": "TestError"}
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = telemetry.send_crash_webhook(report, webhook_url="https://example.com/webhook")
        assert res is True
        mock_post.assert_called_once_with("https://example.com/webhook", json=report, timeout=5)


def test_send_crash_webhook_env_fallback(monkeypatch):
    monkeypatch.setenv("HALGOM_WEBHOOK_URL", "https://env.example.com/webhook")
    report = {"exception_type": "TestError"}
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = telemetry.send_crash_webhook(report)
        assert res is True
        mock_post.assert_called_once_with("https://env.example.com/webhook", json=report, timeout=5)


def test_send_crash_webhook_http_error(caplog):
    report = {"exception_type": "TestError"}
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("requests.post", return_value=mock_resp):
        with caplog.at_level(logging.WARNING):
            res = telemetry.send_crash_webhook(report, webhook_url="https://example.com/webhook")
            assert res is False
            assert "returned non-2xx status code: 500" in caplog.text


def test_send_crash_webhook_network_exception(caplog):
    report = {"exception_type": "TestError"}

    with patch("requests.post", side_effect=requests.RequestException("Connection refused")):
        with caplog.at_level(logging.WARNING):
            res = telemetry.send_crash_webhook(report, webhook_url="https://example.com/webhook")
            assert res is False
            assert "Failed to send crash report webhook" in caplog.text
