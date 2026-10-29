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


# ────────────────────────── compute_sector_rotation tests ──────────────────────────


def test_compute_sector_rotation_full_universe():
    """Verify sector rotation computation on a multi-sector universe with full return columns."""
    df = pd.DataFrame([
        {"ticker": "A", "sector": "Technology", "change": 2.0, "perf_w": 3.0, "perf_m": 5.0, "perf_q": 10.0, "perf_h": 15.0, "price": 100.0, "sma50": 90.0},
        {"ticker": "B", "sector": "Technology", "change": 1.0, "perf_w": 2.0, "perf_m": 4.0, "perf_q": 8.0, "perf_h": 12.0, "price": 100.0, "sma50": 90.0},
        {"ticker": "C", "sector": "Financials", "change": 0.5, "perf_w": 1.0, "perf_m": 2.0, "perf_q": 4.0, "perf_h": 6.0, "price": 50.0, "sma50": 45.0},
        {"ticker": "D", "sector": "Utilities", "change": -1.0, "perf_w": -2.0, "perf_m": -3.0, "perf_q": -5.0, "perf_h": -8.0, "price": 30.0, "sma50": 35.0},
    ])

    res = sector_rotation.compute_sector_rotation(df)
    assert not res.empty
    assert len(res) == 3  # 3 sectors
    assert list(res["sector"]) == ["Technology", "Financials", "Utilities"]

    # Technology is rank 1 on all timeframes, composite_score = 1.0, composite_rank = 1
    tech = res[res["sector"] == "Technology"].iloc[0]
    assert tech["rank_change"] == 1
    assert tech["rank_perf_w"] == 1
    assert tech["rank_perf_m"] == 1
    assert tech["rank_perf_q"] == 1
    assert tech["rank_perf_h"] == 1
    assert tech["composite_score"] == pytest.approx(1.0)
    assert tech["composite_rank"] == 1

    # Utilities is rank 3 on all timeframes, composite_score = 3.0, composite_rank = 3
    util = res[res["sector"] == "Utilities"].iloc[0]
    assert util["composite_score"] == pytest.approx(3.0)
    assert util["composite_rank"] == 3


def test_compute_sector_rotation_single_sector():
    """Single sector universe returns 1-row leaderboard with rank 1."""
    df = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Technology", "change": 1.5, "perf_w": 2.0, "perf_m": 3.0, "perf_q": 5.0, "perf_h": 10.0, "price": 150.0, "sma50": 140.0},
        {"ticker": "MSFT", "sector": "Technology", "change": 0.5, "perf_w": 1.0, "perf_m": 2.0, "perf_q": 4.0, "perf_h": 8.0, "price": 250.0, "sma50": 240.0},
    ])

    res = sector_rotation.compute_sector_rotation(df)
    assert not res.empty
    assert len(res) == 1
    assert res.iloc[0]["sector"] == "Technology"
    assert res.iloc[0]["composite_score"] == pytest.approx(1.0)
    assert res.iloc[0]["composite_rank"] == 1


def test_compute_sector_rotation_missing_columns():
    """Handles missing return columns, empty universe, and universe with no sector data gracefully."""
    # Universe with only 'change' and 'perf_m' available
    df_partial = pd.DataFrame([
        {"ticker": "A", "sector": "Energy", "change": 3.0, "perf_m": 5.0, "price": 50.0, "sma50": 45.0},
        {"ticker": "B", "sector": "Health Care", "change": 1.0, "perf_m": 2.0, "price": 80.0, "sma50": 75.0},
    ])
    res_partial = sector_rotation.compute_sector_rotation(df_partial)
    assert not res_partial.empty
    assert len(res_partial) == 2
    assert "rank_change" in res_partial.columns
    assert "rank_perf_m" in res_partial.columns
    assert "rank_perf_w" not in res_partial.columns
    assert res_partial.iloc[0]["sector"] == "Energy"

    # Universe with no performance columns at all
    df_no_perf = pd.DataFrame([
        {"ticker": "A", "sector": "Energy", "price": 50.0},
        {"ticker": "B", "sector": "Health Care", "price": 80.0},
    ])
    res_no_perf = sector_rotation.compute_sector_rotation(df_no_perf)
    assert res_no_perf.empty

    # Empty universe
    res_empty = sector_rotation.compute_sector_rotation(pd.DataFrame())
    assert res_empty.empty


def test_ui_render_error():
    """Error path logs and shows hebrew_note(t('Error displaying sector rotation'))."""
    # Passing None to sector_rotation_view.render inside _script_render_error triggers exception inside try/except block of render()
    at = AppTest.from_function(_script_render_error, args=(None,)).run()
    assert not at.exception

    # Check that error note is displayed
    error_text = i18n.ui("Error displaying sector rotation", i18n.HE)
    has_err_note = any(error_text in str(html.value) for html in at.html)
    assert has_err_note
