"""Tests for screen diff and movers calculation functions."""
import math
import numpy as np
import pandas as pd
import pytest

from screener.screen_diff import diff, movers


def test_diff_small_frames():
    old = pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOGL"]})
    new = pd.DataFrame({"ticker": ["MSFT", "GOOGL", "AMZN", "NVDA"]})

    res = diff(old, new)
    assert res == {
        "entered": ["AMZN", "NVDA"],
        "left": ["AAPL"],
        "stayed": ["GOOGL", "MSFT"],
        "n_old": 3,
        "n_new": 4,
    }


def test_diff_duplicates_and_mixed_case():
    old = pd.DataFrame({"ticker": [" aapl ", "AAPL", "msft", "MSFT ", np.nan, ""]})
    new = pd.DataFrame({"ticker": ["Aapl", "tsla ", "  TSLA", None]})

    res = diff(old, new)
    assert res == {
        "entered": ["TSLA"],
        "left": ["MSFT"],
        "stayed": ["AAPL"],
        "n_old": 2,
        "n_new": 2,
    }


def test_diff_empty_and_missing_key():
    old = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})
    empty_df = pd.DataFrame()
    no_key_df = pd.DataFrame({"symbol": ["AAPL", "MSFT"]})

    # Empty new
    res_empty_new = diff(old, empty_df)
    assert res_empty_new == {
        "entered": [],
        "left": ["AAPL", "MSFT"],
        "stayed": [],
        "n_old": 2,
        "n_new": 0,
    }

    # Empty old
    res_empty_old = diff(empty_df, old)
    assert res_empty_old == {
        "entered": ["AAPL", "MSFT"],
        "left": [],
        "stayed": [],
        "n_old": 0,
        "n_new": 2,
    }

    # Missing key column treated as empty set
    res_no_key = diff(old, no_key_df)
    assert res_no_key == {
        "entered": [],
        "left": ["AAPL", "MSFT"],
        "stayed": [],
        "n_old": 2,
        "n_new": 0,
    }


def test_diff_custom_key():
    old = pd.DataFrame({"code": ["a", "b"]})
    new = pd.DataFrame({"code": ["b", "c"]})

    res = diff(old, new, key="code")
    assert res == {
        "entered": ["C"],
        "left": ["A"],
        "stayed": ["B"],
        "n_old": 2,
        "n_new": 2,
    }


def test_movers_basic_ordering_and_top_n():
    old = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "price": [100.0, 200.0, 150.0, 300.0],
    })
    new = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "price": [105.0, 180.0, 150.0, 350.0],  # changes: +5, -20, 0, +50
    })

    res = movers(old, new, col="price", top_n=2)

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ["ticker", "old", "new", "change"]
    assert len(res) == 2

    # AMZN (+50, abs 50), MSFT (-20, abs 20)
    assert res.iloc[0]["ticker"] == "AMZN"
    assert res.iloc[0]["old"] == 300.0
    assert res.iloc[0]["new"] == 350.0
    assert res.iloc[0]["change"] == 50.0

    assert res.iloc[1]["ticker"] == "MSFT"
    assert res.iloc[1]["old"] == 200.0
    assert res.iloc[1]["new"] == 180.0
    assert res.iloc[1]["change"] == -20.0


def test_movers_nan_and_invalid_handling():
    old = pd.DataFrame({
        "ticker": ["AAPL", "NVDA", "TSLA", "META", "GOOGL"],
        "pe": [20.0, np.nan, 50.0, "invalid", 15.0],
    })
    new = pd.DataFrame({
        "ticker": ["AAPL", "NVDA", "TSLA", "META", "GOOGL"],
        "pe": [25.0, 40.0, np.nan, 30.0, 18.0],
    })

    res = movers(old, new, col="pe")

    # AAPL (20->25, +5) and GOOGL (15->18, +3) are valid in both
    # NVDA: old is NaN -> dropped
    # TSLA: new is NaN -> dropped
    # META: old is "invalid" -> dropped
    assert len(res) == 2
    assert list(res["ticker"]) == ["AAPL", "GOOGL"]
    assert res.iloc[0]["change"] == 5.0
    assert res.iloc[1]["change"] == 3.0


def test_movers_missing_columns_and_empty():
    old = pd.DataFrame({"ticker": ["AAPL"], "price": [100.0]})
    new = pd.DataFrame({"ticker": ["AAPL"], "volume": [1000]})

    # Column "price" missing in new
    res = movers(old, new, col="price")
    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ["ticker", "old", "new", "change"]
    assert len(res) == 0

    # Key column missing
    res_nokey = movers(old, new, col="price", key="missing")
    assert len(res_nokey) == 0


def test_movers_duplicates_case_and_ties():
    old = pd.DataFrame({
        "ticker": [" aapl ", "AAPL", "googl"],
        "val": [10.0, 999.0, 20.0],
    })
    new = pd.DataFrame({
        "ticker": ["Aapl", "GOOGL"],
        "val": [15.0, 15.0],  # AAPL: 10->15 (+5), GOOGL: 20->15 (-5)
    })

    res = movers(old, new, col="val")

    assert len(res) == 2
    # Absolute change for both is 5.0. Tie-breaker is ticker ascending: AAPL before GOOGL.
    assert list(res["ticker"]) == ["AAPL", "GOOGL"]
    assert res.iloc[0]["change"] == 5.0
    assert res.iloc[1]["change"] == -5.0
