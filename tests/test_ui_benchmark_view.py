"""Tests for screener/ui/benchmark_view.py component using Streamlit AppTest.
"""
from __future__ import annotations

import math
import pytest
from streamlit.testing.v1 import AppTest


def test_benchmark_view_full_stats():
    stats = {
        "n_obs": 120,
        "beta": 1.15,
        "alpha_annual": 4.50,
        "correlation": 0.88,
        "tracking_error": 6.20,
        "up_capture": 1.10,
        "down_capture": 0.95,
    }

    def script(s):
        from screener.ui.benchmark_view import render
        render(s, benchmark_name="SPY")

    at = AppTest.from_function(script, args=(stats,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    metrics = at.metric
    assert len(metrics) == 6

    m_beta = [m for m in metrics if m.label == "בטא"][0]
    m_alpha = [m for m in metrics if m.label == "אלפא שנתית"][0]
    m_corr = [m for m in metrics if m.label == "מתאם"][0]
    m_te = [m for m in metrics if m.label == "שגיאת מעקב"][0]
    m_up = [m for m in metrics if m.label == "תפיסת עליות"][0]
    m_down = [m for m in metrics if m.label == "תפיסת ירידות"][0]

    assert m_beta.value == "1.15"
    assert m_alpha.value == "4.50%"
    assert m_corr.value == "0.88"
    assert m_te.value == "6.20%"
    assert m_up.value == "1.10"
    assert m_down.value == "0.95%"

    # Verify no 'None' or 'nan' text in any metric
    for m in metrics:
        val = str(m.value)
        assert "None" not in val
        assert "nan" not in val.lower()

    # Check caption
    captions = at.caption
    assert any("בטא 1" in c.value and "SPY" in c.value for c in captions)


def test_benchmark_view_all_none_stats():
    stats = {
        "n_obs": 100,
        "beta": None,
        "alpha_annual": None,
        "correlation": None,
        "tracking_error": None,
        "up_capture": None,
        "down_capture": None,
    }

    def script(s):
        from screener.ui.benchmark_view import render
        render(s)

    at = AppTest.from_function(script, args=(stats,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    metrics = at.metric
    assert len(metrics) == 6

    for m in metrics:
        assert m.value == "—"
        assert "None" not in str(m.value)
        assert "nan" not in str(m.value).lower()


def test_benchmark_view_nan_stats():
    stats = {
        "n_obs": 100,
        "beta": float("nan"),
        "alpha_annual": float("nan"),
        "correlation": float("nan"),
        "tracking_error": float("nan"),
        "up_capture": float("nan"),
        "down_capture": float("nan"),
    }

    def script(s):
        from screener.ui.benchmark_view import render
        render(s)

    at = AppTest.from_function(script, args=(stats,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    metrics = at.metric
    assert len(metrics) == 6

    for m in metrics:
        assert m.value == "—"
        assert "None" not in str(m.value)
        assert "nan" not in str(m.value).lower()


def test_benchmark_view_none_input():
    def script(s):
        from screener.ui.benchmark_view import render
        render(s)

    at = AppTest.from_function(script, args=(None,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    info_boxes = at.info
    assert any("אין מספיק נתונים להשוואה" in box.value for box in info_boxes)


def test_benchmark_view_empty_and_zero_n_obs():
    def script_empty(s):
        from screener.ui.benchmark_view import render
        render(s)

    at_empty = AppTest.from_function(script_empty, args=({},)).run()
    assert not at_empty.exception
    assert any("אין מספיק נתונים להשוואה" in box.value for box in at_empty.info)

    def script_zero(s):
        from screener.ui.benchmark_view import render
        render(s)

    at_zero = AppTest.from_function(script_zero, args=({"n_obs": 0},)).run()
    assert not at_zero.exception
    assert any("אין מספיק נתונים להשוואה" in box.value for box in at_zero.info)


def test_benchmark_view_short_sample_warning():
    stats = {
        "n_obs": 30,
        "beta": 0.95,
        "alpha_annual": 2.10,
        "correlation": 0.75,
        "tracking_error": 5.10,
        "up_capture": 0.90,
        "down_capture": 0.85,
    }

    def script(s):
        from screener.ui.benchmark_view import render
        render(s, benchmark_name="QQQ")

    at = AppTest.from_function(script, args=(stats,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Warning for short sample present
    warnings = at.warning
    assert any("מדגם קצר" in w.value for w in warnings)

    # Metrics still rendered
    metrics = at.metric
    assert len(metrics) == 6

    # Custom benchmark name in caption
    captions = at.caption
    assert any("QQQ" in c.value for c in captions)


def test_benchmark_view_error_handling():
    stats = {"n_obs": 100}

    def script(s):
        from screener.ui import benchmark_view
        old_st = benchmark_view.st
        try:
            benchmark_view.st = None  # force exception
            benchmark_view.render(s)
        finally:
            benchmark_view.st = old_st

    at = AppTest.from_function(script, args=(stats,)).run()
    assert not at.exception
