"""
Tests for screener/sector_rotation.py and screener/ui/sector_rotation_view.py.
"""
from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from screener import i18n, sector_rotation


# ────────────────────────── Pure rotation_table tests ──────────────────────────


def test_rotation_table_four_quadrants():
    """Construct a 4-sector DataFrame where each sector falls into a distinct quadrant."""
    # We want benchmark_q = 0.0 and benchmark_m = 0.0 (equal weighting)
    # Sector 1 (Leading): perf_q = 3.0, perf_m = 2.0 -> rs_trend = 3.0 (>=0), rs_mom = (2.0 - 0) - (3.0/3.0) = 1.0 (>=0)
    # Sector 2 (Weakening): perf_q = 3.0, perf_m = 0.0 -> rs_trend = 3.0 (>=0), rs_mom = (0.0 - 0) - (3.0/3.0) = -1.0 (<0)
    # Sector 3 (Lagging): perf_q = -3.0, perf_m = -2.0 -> rs_trend = -3.0 (<0), rs_mom = (-2.0 - 0) - (-3.0/3.0) = -1.0 (<0)
    # Sector 4 (Improving): perf_q = -3.0, perf_m = 0.0 -> rs_trend = -3.0 (<0), rs_mom = (0.0 - 0) - (-3.0/3.0) = 1.0 (>=0)

    df = pd.DataFrame([
        {"ticker": "A", "sector": "Technology", "perf_q": 3.0, "perf_m": 2.0, "market_cap": 100.0, "price": 10.0, "sma50": 9.0},
        {"ticker": "B", "sector": "Finance", "perf_q": 3.0, "perf_m": 0.0, "market_cap": 100.0, "price": 10.0, "sma50": 9.0},
        {"ticker": "C", "sector": "Utilities", "perf_q": -3.0, "perf_m": -2.0, "market_cap": 100.0, "price": 10.0, "sma50": 9.0},
        {"ticker": "D", "sector": "Energy", "perf_q": -3.0, "perf_m": 0.0, "market_cap": 100.0, "price": 10.0, "sma50": 9.0},
    ])

    rot = sector_rotation.rotation_table(df, weight="equal")

    assert len(rot) == 4
    assert list(rot.columns) == sector_rotation.ROTATION_COLUMNS

    tech = rot[rot["sector"] == "Technology"].iloc[0]
    assert tech["rs_trend"] == pytest.approx(3.0)
    assert tech["rs_momentum"] == pytest.approx(1.0)
    assert tech["quadrant"] == "Leading"
    assert tech["quadrant_he"] == "מובילים"

    fin = rot[rot["sector"] == "Finance"].iloc[0]
    assert fin["rs_trend"] == pytest.approx(3.0)
    assert fin["rs_momentum"] == pytest.approx(-1.0)
    assert fin["quadrant"] == "Weakening"
    assert fin["quadrant_he"] == "נחלשים"

    util = rot[rot["sector"] == "Utilities"].iloc[0]
    assert util["rs_trend"] == pytest.approx(-3.0)
    assert util["rs_momentum"] == pytest.approx(-1.0)
    assert util["quadrant"] == "Lagging"
    assert util["quadrant_he"] == "מפגרים"

    nrg = rot[rot["sector"] == "Energy"].iloc[0]
    assert nrg["rs_trend"] == pytest.approx(-3.0)
    assert nrg["rs_momentum"] == pytest.approx(1.0)
    assert nrg["quadrant"] == "Improving"
    assert nrg["quadrant_he"] == "משתפרים"


def test_rotation_table_empty():
    """Empty DataFrame returns empty DataFrame with ROTATION_COLUMNS."""
    empty_df = pd.DataFrame()
    rot = sector_rotation.rotation_table(empty_df)
    assert rot.empty
    assert list(rot.columns) == sector_rotation.ROTATION_COLUMNS

    df_missing = pd.DataFrame([{"ticker": "A", "price": 10.0}])
    rot2 = sector_rotation.rotation_table(df_missing)
    assert rot2.empty
    assert list(rot2.columns) == sector_rotation.ROTATION_COLUMNS


# ────────────────────────── AppTest UI tests ──────────────────────────


def _script_render_normal(df):
    from screener.ui import sector_rotation_view
    sector_rotation_view.render(df)


def _script_render_error(df):
    from screener.ui import sector_rotation_view
    sector_rotation_view.render(df)


def test_ui_render_normal():
    """Renders sr_weight, sr_chart, and sr_table with no exception."""
    df = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Technology", "perf_q": 5.0, "perf_m": 2.0, "market_cap": 100.0, "price": 150.0, "sma50": 140.0},
        {"ticker": "JPM", "sector": "Finance", "perf_q": -2.0, "perf_m": -1.0, "market_cap": 80.0, "price": 100.0, "sma50": 105.0},
    ])

    at = AppTest.from_function(_script_render_normal, args=(df,)).run()
    assert not at.exception
    # Check widgets
    assert len(at.segmented_control) >= 1
    assert at.segmented_control[0].key == "sr_weight"

    charts = at.get("plotly_chart")
    assert len(charts) >= 1

    tables = at.get("dataframe")
    assert len(tables) >= 1


def test_ui_render_empty():
    """Empty universe shows t('No sector data')."""
    empty_df = pd.DataFrame()
    at = AppTest.from_function(_script_render_normal, args=(empty_df,)).run()
    assert not at.exception
    assert len(at.segmented_control) >= 1
    assert at.segmented_control[0].key == "sr_weight"

    # Should show hebrew_note for "No sector data" ("אין נתוני סקטורים")
    hebrew_text = i18n.ui("No sector data", i18n.HE)
    has_note = any(hebrew_text in str(html.value) for html in at.html)
    assert has_note


def test_ui_render_error():
    """Error path logs and shows hebrew_note(t('Error displaying sector rotation'))."""
    # Passing None to sector_rotation_view.render inside _script_render_error triggers exception inside try/except block of render()
    at = AppTest.from_function(_script_render_error, args=(None,)).run()
    assert not at.exception

    # Check that error note is displayed
    error_text = i18n.ui("Error displaying sector rotation", i18n.HE)
    has_err_note = any(error_text in str(html.value) for html in at.html)
    assert has_err_note
