"""Tests for ATR position sizer and trailing stop calculator (screener/sizing.py)."""
import math
import numpy as np
import pandas as pd
import pytest

from screener.portfolio import PORTFOLIO, WATCHLIST, PList
from screener.sizing import STOP_COLUMNS, position_plan, stop_warnings, trailing_stops


def test_position_plan_exact():
    res = position_plan(
        price=100.0,
        atr=2.5,
        account_size=50000.0,
        risk_pct=1.0,
        atr_mult=2.0,
        target_r=2.0,
    )
    assert res is not None
    assert res["shares"] == 100
    assert res["position_value"] == 10000.0
    assert res["position_weight_pct"] == 20.0
    assert res["stop_price"] == 95.0
    assert res["stop_pct"] == -5.0
    assert res["target_price"] == 110.0
    assert res["target_pct"] == 10.0
    assert res["risk_dollars"] == 500.0
    assert res["actual_risk_dollars"] == 500.0
    assert res["reward_dollars"] == 1000.0
    assert "קנייה של 100 מניות ($10,000)" in res["summary_he"]
    assert "סטופ ב-$95.00 (-5.0%)" in res["summary_he"]
    assert "סיכון $500" in res["summary_he"]


def test_position_plan_share_capping():
    # Tiny ATR implies high raw_shares:
    # risk_dollars = 10000 * 0.01 = 100
    # stop_distance = 0.1 * 2 = 0.2
    # raw_shares = 100 / 0.2 = 500 shares
    # max_shares by cash = 10000 / 100 = 100 shares
    res = position_plan(
        price=100.0,
        atr=0.1,
        account_size=10000.0,
        risk_pct=1.0,
        atr_mult=2.0,
    )
    assert res is not None
    assert res["shares"] == 100
    assert res["position_value"] == 10000.0
    assert res["position_weight_pct"] == 100.0
    assert res["actual_risk_dollars"] == 20.0


def test_position_plan_invalid_inputs():
    # NaN inputs
    assert position_plan(np.nan, 2.5, 50000.0) is None
    assert position_plan(100.0, np.nan, 50000.0) is None
    assert position_plan(100.0, 2.5, np.nan) is None
    assert position_plan(100.0, 2.5, 50000.0, risk_pct=np.nan) is None

    # Zero or negative inputs
    assert position_plan(0.0, 2.5, 50000.0) is None
    assert position_plan(-100.0, 2.5, 50000.0) is None
    assert position_plan(100.0, 0.0, 50000.0) is None
    assert position_plan(100.0, 2.5, 0.0) is None
    assert position_plan(100.0, 2.5, 50000.0, risk_pct=0.0) is None
    assert position_plan(100.0, 2.5, 50000.0, atr_mult=0.0) is None
    assert position_plan(100.0, 2.5, 50000.0, target_r=-1.0) is None

    # atr * atr_mult >= price
    assert position_plan(10.0, 5.0, 50000.0, atr_mult=2.0) is None  # 5*2 = 10 >= 10
    assert position_plan(10.0, 6.0, 50000.0, atr_mult=2.0) is None  # 6*2 = 12 > 10


def test_trailing_stops_and_warnings():
    # Build a portfolio with open and closed positions
    plist = PList(
        name="Test Portfolio",
        kind=PORTFOLIO,
        rows=[
            # AAPL: open position (bought 10, sold 0), price <= stop_price -> "hit"
            # avg_cost=100, price=90, high_1m=110 -> ref_high=110, stop_price=110 - 2*10 = 90. price(90) <= stop_price(90) -> hit
            {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},

            # MSFT: open position, "near"
            # avg_cost=200, price=201, high_1m=205 -> ref_high=205, stop_price=205 - 2*2.5 = 200.
            # distance_pct = (201 - 200) / 201 * 100 = 0.497% <= 1.5% -> near
            {"ticker": "MSFT", "date": "2025-01-01", "price": 200.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},

            # NVDA: open position, "ok"
            # avg_cost=100, price=130, high_1m=130 -> ref_high=130, stop_price=130 - 2*2 = 126.
            # distance_pct = (130 - 126) / 130 * 100 = 3.07% > 1.5% -> ok
            {"ticker": "NVDA", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},

            # TSLA: open position, "no_data" (missing ATR)
            {"ticker": "TSLA", "date": "2025-01-01", "price": 200.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},

            # AMD: closed position (bought 10, sold 10) -> quantity == 0, excluded
            {"ticker": "AMD", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
            {"ticker": "AMD", "date": "2025-01-02", "price": 105.0, "quantity": 10, "side": "sell", "fees": 0.0, "note": ""},
        ],
    )

    universe = pd.DataFrame([
        {"ticker": "AAPL", "company": "Apple Inc.", "price": 90.0, "atr": 10.0, "high_1m": 110.0},
        {"ticker": "MSFT", "company": "Microsoft Corp.", "price": 201.0, "atr": 2.5, "high_1m": 205.0},
        {"ticker": "NVDA", "company": "NVIDIA Corp.", "price": 130.0, "atr": 2.0, "high_1m": 130.0},
        {"ticker": "TSLA", "company": "Tesla Inc.", "price": 220.0, "atr": np.nan, "high_1m": 230.0},
        {"ticker": "AMD", "company": "Advanced Micro Devices", "price": 150.0, "atr": 5.0, "high_1m": 160.0},
    ])

    df = trailing_stops(plist, universe, atr_mult=2.0, warn_within_pct=1.5)

    assert list(df.columns) == STOP_COLUMNS
    # AMD excluded because quantity == 0
    assert set(df["ticker"]) == {"AAPL", "MSFT", "NVDA", "TSLA"}

    # Order must be "hit" -> "near" -> "ok" -> "no_data"
    statuses = df["status"].tolist()
    assert statuses == ["hit", "near", "ok", "no_data"]

    # Check AAPL hit
    aapl_row = df[df["ticker"] == "AAPL"].iloc[0]
    assert aapl_row["status"] == "hit"
    assert aapl_row["ref_high"] == 110.0
    assert aapl_row["stop_price"] == 90.0
    assert aapl_row["distance_pct"] == 0.0

    # Check TSLA no_data
    tsla_row = df[df["ticker"] == "TSLA"].iloc[0]
    assert tsla_row["status"] == "no_data"
    assert pd.isna(tsla_row["stop_price"])
    assert pd.isna(tsla_row["distance_pct"])

    # Test stop_warnings
    warns = stop_warnings(df)
    assert len(warns) == 2
    assert "AAPL: המחיר ($90.00) חצה את הסטופ הנגרר ($90.00)" in warns[0]
    assert "MSFT: קרובה לסטופ הנגרר ($200.00, מרחק 0.5%)" in warns[1]


def test_trailing_stops_watchlist_and_empty():
    wlist = PList(name="Watchlist", kind=WATCHLIST, rows=[{"ticker": "AAPL", "date": "2025-01-01", "price": 100.0}])
    universe = pd.DataFrame([{"ticker": "AAPL", "company": "Apple", "price": 100.0, "atr": 2.0}])

    df_w = trailing_stops(wlist, universe)
    assert df_w.empty
    assert list(df_w.columns) == STOP_COLUMNS

    empty_p = PList(name="Empty Portfolio", kind=PORTFOLIO, rows=[])
    df_e = trailing_stops(empty_p, universe)
    assert df_e.empty
    assert list(df_e.columns) == STOP_COLUMNS

    assert stop_warnings(df_w) == []


def test_hebrew_strings_no_nan_or_none():
    plan = position_plan(100.0, 2.5, 50000.0)
    assert plan is not None
    assert "nan" not in plan["summary_he"].lower()
    assert "none" not in plan["summary_he"].lower()

    plist = PList(
        name="P",
        kind=PORTFOLIO,
        rows=[
            {"ticker": "AAPL", "date": "2025-01-01", "price": 100.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
            {"ticker": "MSFT", "date": "2025-01-01", "price": 200.0, "quantity": 10, "side": "buy", "fees": 0.0, "note": ""},
        ],
    )
    universe = pd.DataFrame([
        {"ticker": "AAPL", "company": "Apple", "price": 90.0, "atr": 10.0, "high_1m": 110.0},
        {"ticker": "MSFT", "company": "Microsoft", "price": 201.0, "atr": 2.5, "high_1m": 205.0},
    ])
    df = trailing_stops(plist, universe)
    warns = stop_warnings(df)
    for w in warns:
        assert "nan" not in w.lower()
        assert "none" not in w.lower()
