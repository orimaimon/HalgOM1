"""
Tests for load_filtered_universe and DuckDB predicate pushdown in screener.derive and screener.data.derive.
"""
import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from screener.data.derive import load_filtered_universe as load_filtered_data
from screener.derive import _build_where_clause, load_filtered_universe


@pytest.fixture
def sample_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "PENNY", "SPAC", "NULL_STOCK"],
            "company": ["Apple Inc", "Microsoft Corp", "Penny Stock Co", "Acquisition Corp", "Missing Data Inc"],
            "price": [150.50, 300.25, 0.50, 10.00, np.nan],
            "avg_volume": [1000000.0, 2000000.0, 50000.0, 80000.0, 1000.0],
            "market_cap": [2.5e12, 2.8e12, 1e6, 1.5e8, np.nan],
            "pe": [28.5, 32.0, np.nan, 12.0, -5.0],
            "eps_next_fy": [6.0, 10.0, -0.1, 0.5, np.nan],
            "sma20": [148.0, 295.0, 0.48, 9.9, np.nan],
            "close_w": [151.0, 302.0, 0.52, 10.1, np.nan],
            "sma50_w": [140.0, 280.0, 0.40, 9.5, np.nan],
            "high52": [180.0, 350.0, 1.20, 11.0, np.nan],
            "low52": [120.0, 240.0, 0.30, 9.8, np.nan],
            "high_1m": [155.0, 310.0, 0.60, 10.2, np.nan],
            "low_1m": [145.0, 290.0, 0.45, 9.8, np.nan],
            "high_5d": [152.0, 305.0, 0.55, 10.1, np.nan],
            "low_5d": [149.0, 298.0, 0.48, 9.9, np.nan],
            "target": [170.0, 340.0, 1.00, 11.5, np.nan],
            "sector": ["Technology", "Technology", "Basic Materials", "Financial Conglomerates", None],
            "industry": ["Consumer Electronics", "Software", "Mining", "Financial Conglomerates", None],
            "earnings_date": pd.to_datetime(["2026-11-01", "2026-11-05", None, "2026-10-30", None]),
            "is_primary": [True, True, False, True, False],
        }
    )
    p = tmp_path / "universe.parquet"
    df.to_parquet(p, index=False)
    return p, df


def test_module_exports():
    assert load_filtered_universe is load_filtered_data


def test_unfiltered_load_parity(sample_parquet):
    path, raw_df = sample_parquet
    duck_df = load_filtered_universe(path)
    pd_df = pd.read_parquet(path)
    pdt.assert_frame_equal(duck_df, pd_df)


def test_string_where_clause_parity(sample_parquet):
    path, raw_df = sample_parquet

    # Filter: price >= 1.0 AND avg_volume >= 100000
    duck_df = load_filtered_universe(path, where="price >= 1.0 AND avg_volume >= 100000")
    pd_df = raw_df[(raw_df["price"] >= 1.0) & (raw_df["avg_volume"] >= 100000)].reset_index(drop=True)

    pdt.assert_frame_equal(duck_df, pd_df)


def test_list_where_clause_parity(sample_parquet):
    path, raw_df = sample_parquet

    duck_df = load_filtered_universe(path, where=["price >= 5.0", "sector = 'Technology'"])
    pd_df = raw_df[(raw_df["price"] >= 5.0) & (raw_df["sector"] == "Technology")].reset_index(drop=True)

    pdt.assert_frame_equal(duck_df, pd_df)


def test_dict_where_clause_parity(sample_parquet):
    path, raw_df = sample_parquet

    # Test min_ and max_ prefixes, tuple operators, ranges, lists, NULLs
    where_dict = {
        "min_price": 1.0,
        "sector": "Technology",
        "pe": (">=", 20.0),
    }

    duck_df = load_filtered_universe(path, where=where_dict)
    pd_df = raw_df[
        (raw_df["price"] >= 1.0) & (raw_df["sector"] == "Technology") & (raw_df["pe"].notna()) & (raw_df["pe"] >= 20.0)
    ].reset_index(drop=True)

    pdt.assert_frame_equal(duck_df, pd_df)


def test_dict_range_and_in_clause(sample_parquet):
    path, raw_df = sample_parquet

    where_dict = {
        "price": (1.0, 200.0),
        "ticker": ["AAPL", "MSFT", "SPAC"],
    }

    duck_df = load_filtered_universe(path, where=where_dict)
    pd_df = raw_df[
        (raw_df["price"].between(1.0, 200.0)) & (raw_df["ticker"].isin(["AAPL", "MSFT", "SPAC"]))
    ].reset_index(drop=True)

    pdt.assert_frame_equal(duck_df, pd_df)


def test_column_selection(sample_parquet):
    path, raw_df = sample_parquet
    cols = ["ticker", "price", "sector"]

    duck_df = load_filtered_universe(path, where="price >= 1.0", columns=cols)
    pd_df = raw_df[raw_df["price"] >= 1.0][cols].reset_index(drop=True)

    pdt.assert_frame_equal(duck_df, pd_df)


def test_add_derived_cols(sample_parquet):
    path, raw_df = sample_parquet
    today = dt.date(2026, 10, 20)

    duck_df = load_filtered_universe(
        path, where="price >= 1.0", add_derived_cols=True, today=today
    )

    assert "dollar_volume" in duck_df.columns
    assert "fwd_pe" in duck_df.columns
    assert "sma20_pct" in duck_df.columns
    assert "signals" in duck_df.columns
    assert len(duck_df) == 3


def test_build_where_clause_helpers():
    assert _build_where_clause(None) == ("", [])
    assert _build_where_clause("") == ("", [])
    assert _build_where_clause("   ") == ("", [])
    assert _build_where_clause("price > 10") == (" WHERE price > 10", [])

    sql, params = _build_where_clause(["price > 10", "sector = 'Tech'"])
    assert sql == " WHERE price > 10 AND sector = 'Tech'"
    assert params == []

    sql, params = _build_where_clause(
        {
            "min_price": 5.0,
            "max_pe": 30.0,
            "sector": "Technology",
            "industry": None,
            "ticker": ["AAPL", "MSFT"],
        }
    )
    assert sql == ' WHERE "price" >= ? AND "pe" <= ? AND "sector" = ? AND "industry" IS NULL AND "ticker" IN (?, ?)'
    assert params == [5.0, 30.0, "Technology", "AAPL", "MSFT"]


def test_invalid_where_type():
    with pytest.raises(ValueError, match="Unsupported where clause type"):
        _build_where_clause(12345)
