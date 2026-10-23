"""
Tests for industry/sector relative metrics (derive.py: relative_to_groups) and filters.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from screener.derive import add_derived, relative_to_groups
from screener.filters import screen


def test_relative_industry_pe_and_ps():
    # An industry with P/Es 10, 20, 30, 40, 50 (median 30):
    # the 15 stock gets -50, the 45 stock +50; a stock with P/E -5 gets NaN and does not move the median.
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E", "F", "G"],
        "pe": [10.0, 15.0, 20.0, 30.0, 40.0, 45.0, -5.0],
        "ps": [10.0, 15.0, 20.0, 30.0, 40.0, 45.0, -5.0],
        "price": [10.0] * 7,
        "avg_volume": [100000] * 7,
        "industry": ["Tech"] * 7,
        "sector": ["Technology"] * 7,
    })

    res = relative_to_groups(df)

    # Median of positive P/Es (10, 15, 20, 30, 40, 45) is 25.
    # Wait: test requirement says: "An industry with P/Es 10, 20, 30, 40, 50 (median 30): the 15 stock gets -50, the 45 stock +50"
    # Let us build the exact dataset for 10, 20, 30, 40, 50 + 15 + 45 + -5
    df_exact = pd.DataFrame({
        "ticker": ["T10", "T20", "T30", "T40", "T50", "T15", "T45", "TNEG"],
        "pe": [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 45.0, -5.0],
        "ps": [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 45.0, -5.0],
        "industry": ["Tech"] * 8,
        "sector": ["Technology"] * 8,
    })
    # Positive values: [10, 20, 30, 40, 50, 15, 45] -> sorted: 10, 15, 20, 30, 40, 45, 50 -> median = 30!
    res_exact = relative_to_groups(df_exact)

    # 15 stock: (15 / 30 - 1) * 100 = -50
    assert np.isclose(res_exact["pe_vs_industry"].iloc[5], -50.0)
    # 45 stock: (45 / 30 - 1) * 100 = +50
    assert np.isclose(res_exact["pe_vs_industry"].iloc[6], 50.0)
    # P/E -5 stock gets NaN
    assert np.isnan(res_exact["pe_vs_industry"].iloc[7])

    # ps_vs_industry is identical
    assert np.isclose(res_exact["ps_vs_industry"].iloc[5], -50.0)
    assert np.isclose(res_exact["ps_vs_industry"].iloc[6], 50.0)
    assert np.isnan(res_exact["ps_vs_industry"].iloc[7])


def test_relative_fallback_to_sector_and_nan():
    # An industry with only 3 positive P/Es falls back to its sector median;
    # a sector with fewer than 5 as well gives NaN.
    df = pd.DataFrame({
        "ticker": ["A1", "A2", "A3", "B1", "B2", "C1", "C2"],
        "pe": [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0],
        "ps": [10.0, 20.0, 30.0, 40.0, 50.0, 10.0, 20.0],
        "industry": ["IndA", "IndA", "IndA", "IndB", "IndB", "IndC", "IndC"],
        "sector": ["SecX", "SecX", "SecX", "SecX", "SecX", "SecY", "SecY"],
    })
    # SecX has 5 stocks (IndA has 3, IndB has 2). SecX positive P/E count = 5, median = 30.
    # IndA (3 stocks < 5) falls back to SecX median (30).
    # SecY has only 2 stocks (IndC has 2). SecY positive P/E count = 2 < 5 -> yields NaN.

    res = relative_to_groups(df)

    # IndA stocks use SecX median 30
    assert np.isclose(res["pe_vs_industry"].iloc[0], (10.0 / 30.0 - 1) * 100)
    assert np.isclose(res["pe_vs_industry"].iloc[1], (20.0 / 30.0 - 1) * 100)
    assert np.isclose(res["pe_vs_industry"].iloc[2], (30.0 / 30.0 - 1) * 100)

    # IndC stocks in SecY (total 2 < 5) get NaN
    assert np.isnan(res["pe_vs_industry"].iloc[5])
    assert np.isnan(res["pe_vs_industry"].iloc[6])


def test_sales_growth_vs_sector():
    # sales_growth_vs_sector in percentage points (sector median 10, stock 25 -> +15).
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(5)],
        "sales_growth": [0.0, 5.0, 10.0, 15.0, 25.0],
        "sector": ["Sec1"] * 5,
    })
    # Sector median = 10.0. Stock with sales_growth 25.0 gets 25.0 - 10.0 = +15.0.
    res = relative_to_groups(df)
    assert np.isclose(res["sales_growth_vs_sector"].iloc[4], 15.0)


def test_missing_industry_or_sector():
    # NaN industry / sector -> NaN, no exception; add_derived on a frame without those columns still works.
    df_nan_groups = pd.DataFrame({
        "ticker": ["A", "B"],
        "pe": [15.0, 20.0],
        "ps": [2.0, 3.0],
        "sales_growth": [10.0, 20.0],
        "industry": [np.nan, "Tech"],
        "sector": [np.nan, np.nan],
    })
    res = relative_to_groups(df_nan_groups)
    assert np.isnan(res["pe_vs_industry"].iloc[0])
    assert np.isnan(res["pe_vs_industry"].iloc[1])

    # Test add_derived without industry and sector columns
    df_no_cols = pd.DataFrame({
        "ticker": ["A", "B"],
        "price": [10.0, 20.0],
        "avg_volume": [100000, 200000],
    })
    df_derived = add_derived(df_no_cols)
    assert "pe_vs_industry" in df_derived.columns
    assert "ps_vs_industry" in df_derived.columns
    assert "sales_growth_vs_sector" in df_derived.columns
    assert df_derived["pe_vs_industry"].isna().all()


def test_filters_and_screen_relative():
    # Each filter option keeps the right rows and drops NaN; filters.screen(df, {"pe_vs_industry": "20%+ below industry"}) works.
    df = pd.DataFrame({
        "ticker": ["CHEAP50", "CHEAP20", "FAIR", "EXPENSIVE", "NEG_PE", "NO_IND"],
        "pe": [15.0, 21.0, 30.0, 45.0, -5.0, 15.0],
        "price": [10.0] * 6,
        "avg_volume": [100000] * 6,
        "market_cap": [100e6] * 6,
        "industry": ["Tech"] * 5 + [np.nan],
        "sector": ["Technology"] * 5 + [np.nan],
    })
    # Add dummy extra rows to reach >= 5 count in Tech industry with positive P/E
    d_extra = pd.DataFrame({
        "ticker": ["E1", "E2", "E3"],
        "pe": [30.0, 30.0, 30.0],
        "price": [10.0] * 3,
        "avg_volume": [100000] * 3,
        "market_cap": [100e6] * 3,
        "industry": ["Tech"] * 3,
        "sector": ["Technology"] * 3,
    })
    df_full = pd.concat([df, d_extra], ignore_index=True)
    df_derived = add_derived(df_full)

    # Industry median P/E = 30.0
    # CHEAP50: pe=15 -> pe_vs_ind = -50% <= -50
    # CHEAP20: pe=24 -> pe_vs_ind = -20% <= -20
    # FAIR: pe=30 -> pe_vs_ind = 0
    # EXPENSIVE: pe=45 -> pe_vs_ind = +50% > 0

    res_50 = screen(df_derived, {"pe_vs_industry": "50%+ below industry"}, junk=False)
    assert list(res_50["ticker"]) == ["CHEAP50"]

    res_20 = screen(df_derived, {"pe_vs_industry": "20%+ below industry"}, junk=False)
    assert set(res_20["ticker"]) == {"CHEAP50", "CHEAP20"}

    res_below = screen(df_derived, {"pe_vs_industry": "Below industry median"}, junk=False)
    assert set(res_below["ticker"]) == {"CHEAP50", "CHEAP20"}

    res_above = screen(df_derived, {"pe_vs_industry": "Above industry median"}, junk=False)
    assert list(res_above["ticker"]) == ["EXPENSIVE"]


def test_relative_to_groups_speed_benchmark():
    # Speed: 5,000 rows, 120 industries, 11 sectors -> relative_to_groups < 0.5 s.
    np.random.seed(42)
    n = 5000
    industries = [f"Ind_{i}" for i in range(120)]
    sectors = [f"Sec_{i}" for i in range(11)]

    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "pe": np.random.uniform(-10, 100, size=n),
        "ps": np.random.uniform(-5, 20, size=n),
        "sales_growth": np.random.uniform(-30, 100, size=n),
        "industry": np.random.choice(industries, size=n),
        "sector": np.random.choice(sectors, size=n),
    })

    t0 = time.perf_counter()
    res = relative_to_groups(df)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"relative_to_groups took {elapsed:.3f} s, expected < 0.5 s"
    assert len(res["pe_vs_industry"]) == n
