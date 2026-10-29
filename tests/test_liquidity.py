"""
Tests for position liquidity metrics module (screener/liquidity.py).
"""
import numpy as np
import pandas as pd
import pytest

from screener.liquidity import LIQUIDITY_COLUMNS, liquidity_table, summary


def test_liquidity_table_known_position():
    # Known position: AAPL, 1,000 shares, avg_volume 100,000, participation 10% (0.10)
    # pct_of_adv = 1000 / 100000 * 100 = 1.0%
    # days_to_exit = 1000 / (100000 * 0.10) = 0.1 days
    # level = 'high' (days_to_exit <= 1.0)
    holdings = pd.DataFrame([{"ticker": "AAPL", "shares": 1000, "avg_volume": 100000}])
    res = liquidity_table(holdings, participation=0.10)

    assert list(res.columns) == LIQUIDITY_COLUMNS
    assert len(res) == 1
    row = res.iloc[0]
    assert row["ticker"] == "AAPL"
    assert pytest.approx(row["shares"]) == 1000.0
    assert pytest.approx(row["avg_volume"]) == 100000.0
    assert pytest.approx(row["pct_of_adv"]) == 1.0
    assert pytest.approx(row["days_to_exit"]) == 0.1
    assert row["level"] == "high"


def test_level_boundaries():
    # Boundary test with participation = 0.10
    # For days_to_exit = 1.0 -> shares = avg_vol * 0.10 * 1 = 10,000 shares for 100k avg_vol -> level 'high'
    # For days_to_exit = 1.0001 -> shares = 10,001 -> level 'medium'
    # For days_to_exit = 5.0 -> shares = 50,000 -> level 'medium'
    # For days_to_exit = 5.0001 -> shares = 50,001 -> level 'low'
    holdings = pd.DataFrame(
        [
            {"ticker": "HIGH_EXACT", "shares": 10000, "avg_volume": 100000},
            {"ticker": "MED_JUST_ABOVE", "shares": 10001, "avg_volume": 100000},
            {"ticker": "MED_EXACT", "shares": 50000, "avg_volume": 100000},
            {"ticker": "LOW_JUST_ABOVE", "shares": 50001, "avg_volume": 100000},
        ]
    )
    res = liquidity_table(holdings, participation=0.10)

    levels = dict(zip(res["ticker"], res["level"]))
    assert levels["HIGH_EXACT"] == "high"
    assert levels["MED_JUST_ABOVE"] == "medium"
    assert levels["MED_EXACT"] == "medium"
    assert levels["LOW_JUST_ABOVE"] == "low"

    days = dict(zip(res["ticker"], res["days_to_exit"]))
    assert pytest.approx(days["HIGH_EXACT"]) == 1.0
    assert pytest.approx(days["MED_EXACT"]) == 5.0


def test_zero_and_nan_avg_volume():
    holdings = pd.DataFrame(
        [
            {"ticker": "ZERO_VOL", "shares": 100, "avg_volume": 0},
            {"ticker": "NEG_VOL", "shares": 100, "avg_volume": -500},
            {"ticker": "NAN_VOL", "shares": 100, "avg_volume": np.nan},
        ]
    )
    res = liquidity_table(holdings)

    assert len(res) == 3
    for _, row in res.iterrows():
        assert np.isnan(row["pct_of_adv"])
        assert np.isnan(row["days_to_exit"])
        assert row["level"] == "unknown"


def test_empty_frame():
    holdings = pd.DataFrame(columns=["ticker", "shares", "avg_volume"])
    res = liquidity_table(holdings)

    assert list(res.columns) == LIQUIDITY_COLUMNS
    assert res.empty

    s = summary(res)
    assert s == {"n_low": 0, "n_unknown": 0, "worst": None}


def test_invalid_participation_raises():
    holdings = pd.DataFrame([{"ticker": "AAPL", "shares": 100, "avg_volume": 1000}])

    with pytest.raises(ValueError, match="participation must be in"):
        liquidity_table(holdings, participation=0.0)

    with pytest.raises(ValueError, match="participation must be in"):
        liquidity_table(holdings, participation=-0.1)

    with pytest.raises(ValueError, match="participation must be in"):
        liquidity_table(holdings, participation=1.5)


def test_summary_and_worst_ticker():
    # Create holdings with a mix of levels and a clear worst ticker
    holdings = pd.DataFrame(
        [
            {"ticker": "FAST", "shares": 100, "avg_volume": 100000},       # days = 0.01 ('high')
            {"ticker": "MID", "shares": 20000, "avg_volume": 100000},      # days = 2.0 ('medium')
            {"ticker": "SLOW1", "shares": 60000, "avg_volume": 100000},     # days = 6.0 ('low')
            {"ticker": "SLOW2", "shares": 100000, "avg_volume": 100000},    # days = 10.0 ('low', worst!)
            {"ticker": "UNKN", "shares": 100, "avg_volume": 0},             # level = 'unknown'
        ]
    )

    tbl = liquidity_table(holdings, participation=0.10)
    s = summary(tbl)

    assert s["n_low"] == 2
    assert s["n_unknown"] == 1
    assert s["worst"] == "SLOW2"
