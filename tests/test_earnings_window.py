"""
Unit tests for screener/earnings_window.py.
"""
import math
import numpy as np
import pandas as pd
import pytest

from screener.earnings_window import bucket, upcoming


def test_upcoming_boundaries_and_filtering():
    data = {
        "ticker": ["A0", "B7", "C8", "D14", "E15", "FNEG", "GNAN"],
        "days_to_earnings": [0, 7, 8, 14, 15, -1, math.nan],
        "market_cap": [100, 200, 300, 400, 500, 600, 700],
        "sector": ["Tech", "Health", "Tech", "Finance", "Energy", "Tech", "Health"],
    }
    df = pd.DataFrame(data)

    # Default days = 14: include 0, 7, 8, 14; exclude 15, -1, nan
    res14 = upcoming(df, days=14)
    assert set(res14["ticker"]) == {"A0", "B7", "C8", "D14"}
    assert list(res14["ticker"]) == ["A0", "B7", "C8", "D14"]

    # Custom days = 7
    res7 = upcoming(df, days=7)
    assert set(res7["ticker"]) == {"A0", "B7"}

    # Custom days = 0
    res0 = upcoming(df, days=0)
    assert list(res0["ticker"]) == ["A0"]

    # Custom days = 15
    res15 = upcoming(df, days=15)
    assert set(res15["ticker"]) == {"A0", "B7", "C8", "D14", "E15"}


def test_upcoming_negative_days_parameter():
    df = pd.DataFrame({"ticker": ["AAPL"], "days_to_earnings": [5]})
    with pytest.raises(ValueError, match="days must be >= 0"):
        upcoming(df, days=-1)


def test_upcoming_held_flag_mixed_case():
    df = pd.DataFrame(
        {
            "ticker": ["aapl", "MSFT", "GOOGL", "nvda"],
            "days_to_earnings": [1, 2, 3, 4],
        }
    )
    held = {"AAPL", "msft", "AMZN"}

    res = upcoming(df, days=14, held=held)
    assert res.loc[res["ticker"] == "aapl", "held"].values[0] is True or res.loc[res["ticker"] == "aapl", "held"].values[0] == True
    assert res.loc[res["ticker"] == "MSFT", "held"].values[0] == True
    assert res.loc[res["ticker"] == "GOOGL", "held"].values[0] == False
    assert res.loc[res["ticker"] == "nvda", "held"].values[0] == False


def test_upcoming_missing_columns_and_empty_universe():
    # Empty DataFrame
    empty_df = pd.DataFrame()
    res = upcoming(empty_df)
    assert res.empty
    assert list(res.columns) == ["ticker", "days_to_earnings", "held"]

    # Missing days_to_earnings column
    no_days_df = pd.DataFrame({"ticker": ["AAPL"], "market_cap": [1000]})
    res = upcoming(no_days_df)
    assert res.empty
    assert "market_cap" in res.columns
    assert "days_to_earnings" in res.columns

    # Missing ticker column
    no_ticker_df = pd.DataFrame({"days_to_earnings": [5]})
    res = upcoming(no_ticker_df)
    assert res.empty

    # Universe with optional columns market_cap and sector present
    df_with_opts = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "days_to_earnings": [5],
            "market_cap": [3000000],
            "sector": ["Technology"],
        }
    )
    res = upcoming(df_with_opts)
    assert list(res.columns) == ["ticker", "days_to_earnings", "held", "market_cap", "sector"]
    assert res.iloc[0]["market_cap"] == 3000000
    assert res.iloc[0]["sector"] == "Technology"


def test_upcoming_sort_order():
    df = pd.DataFrame(
        {
            "ticker": ["MSFT", "AAPL", "GOOGL", "AMZN", "META"],
            "days_to_earnings": [2, 2, 0, 5, 0],
        }
    )
    res = upcoming(df)
    # Primary sort: days_to_earnings ascending
    # Secondary sort: ticker ascending
    expected_tickers = ["GOOGL", "META", "AAPL", "MSFT", "AMZN"]
    assert list(res["ticker"]) == expected_tickers
    assert list(res["days_to_earnings"]) == [0, 0, 2, 2, 5]


def test_bucket_grouping():
    df = pd.DataFrame(
        {
            "ticker": ["T0", "W1", "W7", "NW8", "NW14", "L15", "L30", "NEG", "NAN"],
            "days_to_earnings": [0, 1, 7, 8, 14, 15, 30, -5, np.nan],
        }
    )
    buckets = bucket(df)

    assert buckets["today"] == ["T0"]
    assert buckets["this_week"] == ["W1", "W7"]
    assert buckets["next_week"] == ["NW8", "NW14"]
    assert buckets["later"] == ["L15", "L30"]


def test_bucket_empty_and_missing_columns():
    assert bucket(pd.DataFrame()) == {"today": [], "this_week": [], "next_week": [], "later": []}
    assert bucket(None) == {"today": [], "this_week": [], "next_week": [], "later": []}
    assert bucket(pd.DataFrame({"ticker": ["A"]})) == {
        "today": [],
        "this_week": [],
        "next_week": [],
        "later": [],
    }
