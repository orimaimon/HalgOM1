import datetime as dt
import numpy as np
import pandas as pd
import pytest

from screener.portfolio import BUY, SELL, PList
from screener.portfolio_ils import ils_positions, realized_ils


def test_ils_positions_single_buy():
    # Buy 10 @ $100 on 2026-10-01 when rate is 3.70
    # Today 2026-10-05: price $110, rate 3.50
    plist = PList(
        name="Test",
        kind="portfolio",
        rows=[
            {
                "ticker": "AAPL",
                "date": dt.date(2026, 10, 1),
                "price": 100.0,
                "quantity": 10.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            }
        ],
    )
    prices = pd.Series({"AAPL": 110.0})
    rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.70,
            dt.date(2026, 10, 5): 3.50,
        }
    )
    today = dt.date(2026, 10, 5)

    df = ils_positions(plist, prices, rates, today)
    assert len(df) == 1
    row = df.iloc[0]

    assert row["ticker"] == "AAPL"
    assert row["quantity"] == 10.0
    assert row["cost_usd"] == 1000.0
    assert row["value_usd"] == 1100.0
    assert row["cost_ils"] == 3700.0
    assert row["value_ils"] == 3850.0
    assert row["pnl_ils"] == 150.0
    assert pytest.approx(row["pnl_ils_pct"], 0.01) == 4.054  # (150/3700)*100
    assert pytest.approx(row["stock_part_ils"], 0.01) == 370.0  # (1100 - 1000) * 3.70
    assert pytest.approx(row["fx_part_ils"], 0.01) == -220.0   # 1100 * (3.50 - 3.70)
    assert pytest.approx(row["stock_part_ils"] + row["fx_part_ils"], 0.01) == row["pnl_ils"]
    assert df.attrs["no_rate"] == []


def test_two_buys_and_partial_sale():
    # Buy 1: 10 @ $100, rate 3.70 -> spent USD 1000, ILS 3700
    # Buy 2: 10 @ $120, rate 3.50 -> spent USD 1200, ILS 4200
    # Total: 20 shares, USD 2200 (avg $110), ILS 7900 (avg 395 ILS/share)
    # Sell: 10 @ $130, rate 3.60 -> remaining: 10 shares, USD 1100, ILS 3950
    # Today: price $130, rate 3.80
    plist = PList(
        name="Test",
        kind="portfolio",
        rows=[
            {
                "ticker": "MSFT",
                "date": dt.date(2026, 10, 1),
                "price": 100.0,
                "quantity": 10.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            },
            {
                "ticker": "MSFT",
                "date": dt.date(2026, 10, 2),
                "price": 120.0,
                "quantity": 10.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            },
            {
                "ticker": "MSFT",
                "date": dt.date(2026, 10, 3),
                "price": 130.0,
                "quantity": 10.0,
                "side": SELL,
                "fees": 0.0,
                "note": "",
            },
        ],
    )
    prices = pd.Series({"MSFT": 130.0})
    rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.70,
            dt.date(2026, 10, 2): 3.50,
            dt.date(2026, 10, 3): 3.60,
            dt.date(2026, 10, 5): 3.80,
        }
    )
    today = dt.date(2026, 10, 5)

    df = ils_positions(plist, prices, rates, today)
    row = df.iloc[0]

    assert row["quantity"] == 10.0
    assert row["cost_usd"] == 1100.0
    assert row["cost_ils"] == 3950.0  # Retains ILS average cost (7900 / 2)
    assert row["value_usd"] == 1300.0
    assert row["value_ils"] == 4940.0  # 1300 * 3.80
    assert row["pnl_ils"] == 990.0
    assert pytest.approx(row["stock_part_ils"] + row["fx_part_ils"], 0.01) == row["pnl_ils"]


def test_saturday_trade_uses_previous_business_day_rate():
    # Trade on Saturday 2026-10-03 (no rate on Saturday, Friday 2026-10-02 rate is 3.60)
    plist = PList(
        name="Test",
        kind="portfolio",
        rows=[
            {
                "ticker": "NVDA",
                "date": dt.date(2026, 10, 3),  # Saturday
                "price": 100.0,
                "quantity": 5.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            }
        ],
    )
    prices = pd.Series({"NVDA": 100.0})
    rates = pd.Series(
        {
            dt.date(2026, 10, 2): 3.60,  # Friday
            dt.date(2026, 10, 5): 3.60,  # Monday
        }
    )
    today = dt.date(2026, 10, 5)

    df = ils_positions(plist, prices, rates, today)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["cost_ils"] == 5.0 * 100.0 * 3.60
    assert df.attrs["no_rate"] == []


def test_missing_rate_yields_nan_and_no_rate_attr():
    # Trade date 2026-09-01 has no rate within 7 days
    plist = PList(
        name="Test",
        kind="portfolio",
        rows=[
            {
                "ticker": "TSLA",
                "date": dt.date(2026, 9, 1),
                "price": 200.0,
                "quantity": 5.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            }
        ],
    )
    prices = pd.Series({"TSLA": 220.0})
    rates = pd.Series({dt.date(2026, 10, 5): 3.50})
    today = dt.date(2026, 10, 5)

    df = ils_positions(plist, prices, rates, today)
    row = df.iloc[0]

    assert pd.isna(row["cost_ils"])
    assert pd.isna(row["value_ils"])
    assert pd.isna(row["pnl_ils"])
    assert "TSLA" in df.attrs["no_rate"]


def test_realized_ils():
    # Buy 10 @ $100 on 2026-10-01 (rate 3.70) -> cost_ils = 3700 (avg 370)
    # Sell 5 @ $120 on 2026-10-02 (rate 3.60) -> proceeds_ils = 5 * 120 * 3.60 = 2160, cost_ils = 5 * 370 = 1850, pnl_ils = 310
    plist = PList(
        name="Test",
        kind="portfolio",
        rows=[
            {
                "ticker": "AMZN",
                "date": dt.date(2026, 10, 1),
                "price": 100.0,
                "quantity": 10.0,
                "side": BUY,
                "fees": 0.0,
                "note": "",
            },
            {
                "ticker": "AMZN",
                "date": dt.date(2026, 10, 2),
                "price": 120.0,
                "quantity": 5.0,
                "side": SELL,
                "fees": 0.0,
                "note": "",
            },
        ],
    )
    rates = pd.Series(
        {
            dt.date(2026, 10, 1): 3.70,
            dt.date(2026, 10, 2): 3.60,
        }
    )

    df_realized = realized_ils(plist, rates)
    assert len(df_realized) == 1
    row = df_realized.iloc[0]

    assert row["ticker"] == "AMZN"
    assert row["date"] == dt.date(2026, 10, 2)
    assert row["quantity"] == 5.0
    assert row["proceeds_ils"] == 2160.0
    assert row["cost_ils"] == 1850.0
    assert row["pnl_ils"] == 310.0
