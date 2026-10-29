"""
Unit tests for screener/portfolio_xirr.py.
"""

import datetime
import math
import pandas as pd
import pytest

from screener.portfolio_xirr import annualised_to_period, flows_from_trades, xirr


def test_single_buy_365_days():
    d1 = datetime.date(2023, 1, 1)
    d2 = datetime.date(2024, 1, 1)
    flows = [(d1, -100.0), (d2, 110.0)]
    rate = xirr(flows)
    assert rate is not None
    assert abs(rate - 0.10) < 1e-6


def test_two_buys_and_partial_sell():
    d1 = datetime.date(2023, 1, 1)
    d2 = datetime.date(2023, 7, 1)
    d3 = datetime.date(2023, 10, 1)
    d4 = datetime.date(2024, 1, 1)
    flows = [
        (d1, -100.0),
        (d2, -50.0),
        (d3, 30.0),
        (d4, 130.0),
    ]
    rate = xirr(flows)
    assert rate is not None
    assert isinstance(rate, float)
    assert rate > 0.0


def test_loss_case_negative():
    d1 = datetime.date(2023, 1, 1)
    d2 = datetime.date(2024, 1, 1)
    flows = [(d1, -100.0), (d2, 80.0)]
    rate = xirr(flows)
    assert rate is not None
    assert abs(rate - (-0.20)) < 1e-6


def test_all_same_sign_returns_none():
    d1 = datetime.date(2023, 1, 1)
    d2 = datetime.date(2024, 1, 1)
    assert xirr([(d1, 100.0), (d2, 110.0)]) is None
    assert xirr([(d1, -100.0), (d2, -110.0)]) is None


def test_one_flow_returns_none():
    d1 = datetime.date(2023, 1, 1)
    assert xirr([(d1, -100.0)]) is None
    assert xirr([]) is None


def test_same_date_flows_return_none():
    d1 = datetime.date(2023, 1, 1)
    flows = [(d1, -100.0), (d1, 110.0)]
    assert xirr(flows) is None


def test_unsorted_input_gives_same_answer_as_sorted():
    d1 = datetime.date(2023, 1, 1)
    d2 = datetime.date(2023, 7, 1)
    d3 = datetime.date(2024, 1, 1)
    flows_sorted = [(d1, -100.0), (d2, -50.0), (d3, 170.0)]
    flows_unsorted = [(d3, 170.0), (d1, -100.0), (d2, -50.0)]

    rate_sorted = xirr(flows_sorted)
    rate_unsorted = xirr(flows_unsorted)

    assert rate_sorted is not None
    assert rate_unsorted is not None
    assert abs(rate_sorted - rate_unsorted) < 1e-9


def test_flows_from_trades_skips_bad_rows():
    d1 = datetime.date(2023, 1, 1)
    d2 = datetime.date(2023, 6, 1)
    end_d = datetime.date(2024, 1, 1)

    trades_data = [
        {"date": d1, "side": "buy", "shares": 10, "price": 10.0},  # valid buy: -100
        {"date": d2, "side": "sell", "shares": 5, "price": 12.0},  # valid sell: +60
        {"date": d1, "side": "buy", "shares": -5, "price": 10.0},  # bad: negative shares
        {"date": d1, "side": "buy", "shares": 5, "price": 0.0},  # bad: zero price
        {"date": d1, "side": "hold", "shares": 5, "price": 10.0},  # bad: invalid side
        {"date": None, "side": "buy", "shares": 5, "price": 10.0},  # bad: missing date
    ]
    df = pd.DataFrame(trades_data)

    flows = flows_from_trades(df, end_value=65.0, end_date=end_d)
    assert len(flows) == 3
    assert flows[0] == (d1, -100.0)
    assert flows[1] == (d2, 60.0)
    assert flows[2] == (end_d, 65.0)


def test_annualised_to_period():
    assert abs(annualised_to_period(0.10, 365) - 0.10) < 1e-9
    assert abs(annualised_to_period(0.0, 180) - 0.0) < 1e-9
    assert math.isnan(annualised_to_period(-1.5, 100))
    assert math.isnan(annualised_to_period(None, 100))
