"""Tests for screener/ui/sector_rotation_view.py component using Streamlit AppTest.
"""
from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def test_render_sector_rotation_empty():
    def script(df):
        from screener.ui.sector_rotation_view import render_sector_rotation
        render_sector_rotation(df)

    # None DataFrame
    at = AppTest.from_function(script, args=(None,)).run()
    assert not at.exception

    # Empty DataFrame
    at = AppTest.from_function(script, args=(pd.DataFrame(),)).run()
    assert not at.exception
    assert len(at.metric) == 0
    assert len(at.get("plotly_chart")) == 0


def test_render_sector_rotation_populated_stock_universe():
    universe = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Electronic Technology", "perf_q": 15.0, "perf_m": 5.0, "market_cap": 3e12},
        {"ticker": "MSFT", "sector": "Electronic Technology", "perf_q": 12.0, "perf_m": 4.0, "market_cap": 2.8e12},
        {"ticker": "XOM", "sector": "Energy Minerals", "perf_q": 20.0, "perf_m": 8.0, "market_cap": 5e11},
        {"ticker": "JNJ", "sector": "Health Technology", "perf_q": -5.0, "perf_m": -2.0, "market_cap": 4e11},
        {"ticker": "PG", "sector": "Consumer Non-Durables", "perf_q": -10.0, "perf_m": -4.0, "market_cap": 3e11},
    ])

    def script(df):
        from screener.ui.sector_rotation_view import render_sector_rotation
        render_sector_rotation(df)

    at = AppTest.from_function(script, args=(universe,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Check top/bottom metric cards
    assert len(at.metric) >= 3

    # Check bar chart and dataframe
    assert len(at.get("plotly_chart")) >= 1
    assert len(at.get("dataframe")) >= 1


def test_render_sector_rotation_pre_aggregated_sector_df():
    sector_df = pd.DataFrame([
        {
            "sector": "Energy Minerals",
            "rs_trend": 10.5,
            "rs_momentum": 2.1,
            "perf_q": 20.0,
            "perf_m": 8.0,
            "quadrant": "Leading",
            "rank_change": 1,
        },
        {
            "sector": "Electronic Technology",
            "rs_trend": 5.2,
            "rs_momentum": 1.0,
            "perf_q": 15.0,
            "perf_m": 5.0,
            "quadrant": "Leading",
            "rank_change": 0,
        },
        {
            "sector": "Health Technology",
            "rs_trend": -4.0,
            "rs_momentum": -1.2,
            "perf_q": -5.0,
            "perf_m": -2.0,
            "quadrant": "Lagging",
            "rank_change": -1,
        },
    ])

    def script(df):
        from screener.ui.sector_rotation_view import render_sector_rotation
        render_sector_rotation(df)

    at = AppTest.from_function(script, args=(sector_df,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    assert len(at.metric) >= 3
    assert len(at.get("plotly_chart")) >= 1
    assert len(at.get("dataframe")) >= 1


def test_render_compatibility():
    universe = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Electronic Technology", "perf_q": 15.0, "perf_m": 5.0, "market_cap": 3e12},
        {"ticker": "XOM", "sector": "Energy Minerals", "perf_q": 20.0, "perf_m": 8.0, "market_cap": 5e11},
    ])

    def script(df):
        from screener.ui.sector_rotation_view import render
        render(df)

    at = AppTest.from_function(script, args=(universe,)).run()

    if at.exception:
        raise at.exception[0]
    assert not at.exception

    # Scatter chart + leaderboard chart
    assert len(at.get("plotly_chart")) >= 2
    assert len(at.get("dataframe")) >= 1


def test_render_sector_rotation_exception_handled():
    def script():
        import pandas as pd
        import screener.sector_rotation as sr

        def failing_table(*args, **kwargs):
            raise RuntimeError("Simulated rotation error")

        sr.rotation_table = failing_table

        from screener.ui.sector_rotation_view import render_sector_rotation
        render_sector_rotation(pd.DataFrame([{"ticker": "AAPL", "sector": "Tech"}]))

    at = AppTest.from_function(script).run()
    assert not at.exception
