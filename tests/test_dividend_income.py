"""
Unit tests for screener/dividend_income.py.
"""

import math
import numpy as np
import pandas as pd
import pytest

from screener.dividend_income import income_summary, projected_income


def test_yield_percent_unit_calculation():
    """
    Explicit test for the 0.75 yield unit convention:
    0.75 yield on 10,000 value gives 75 annual income (not 7,500 or 0.75).
    """
    holdings = pd.DataFrame(
        [
            {"ticker": "MSFT", "shares": 100, "price": 100.0, "div_yield": 0.75},
        ]
    )
    res = projected_income(holdings)

    assert len(res) == 1
    assert res.loc[0, "value"] == 10000.0
    assert res.loc[0, "div_yield"] == 0.75
    assert pytest.approx(res.loc[0, "annual_income"], abs=1e-6) == 75.0
    assert pytest.approx(res.loc[0, "monthly_income"], abs=1e-6) == 6.25
    assert res.loc[0, "yield_on_cost_unknown"] is True or res.loc[0, "yield_on_cost_unknown"] == True


def test_nan_and_negative_yield_handling():
    """
    NaN or negative div_yield counts as 0 income, but the row stays in output.
    """
    holdings = pd.DataFrame(
        [
            {"ticker": "AAPL", "shares": 10, "price": 150.0, "div_yield": float("nan")},
            {"ticker": "TSLA", "shares": 5, "price": 200.0, "div_yield": -1.5},
            {"ticker": "NVDA", "shares": 20, "price": 100.0, "div_yield": 1.2},
        ]
    )
    res = projected_income(holdings)

    assert len(res) == 3
    aapl = res[res["ticker"] == "AAPL"].iloc[0]
    assert math.isnan(aapl["div_yield"])
    assert aapl["annual_income"] == 0.0
    assert aapl["monthly_income"] == 0.0

    tsla = res[res["ticker"] == "TSLA"].iloc[0]
    assert tsla["div_yield"] == -1.5
    assert tsla["annual_income"] == 0.0
    assert tsla["monthly_income"] == 0.0

    nvda = res[res["ticker"] == "NVDA"].iloc[0]
    assert nvda["annual_income"] == 24.0


def test_zero_and_negative_shares_or_price_dropped():
    """
    Rows with NaN or non-positive (<=0) shares or price are dropped.
    """
    holdings = pd.DataFrame(
        [
            {"ticker": "ZERO_SH", "shares": 0, "price": 100.0, "div_yield": 2.0},
            {"ticker": "NEG_SH", "shares": -10, "price": 100.0, "div_yield": 2.0},
            {"ticker": "NAN_SH", "shares": float("nan"), "price": 100.0, "div_yield": 2.0},
            {"ticker": "ZERO_PR", "shares": 10, "price": 0.0, "div_yield": 2.0},
            {"ticker": "NEG_PR", "shares": 10, "price": -50.0, "div_yield": 2.0},
            {"ticker": "NAN_PR", "shares": 10, "price": float("nan"), "div_yield": 2.0},
            {"ticker": "VALID", "shares": 10, "price": 100.0, "div_yield": 2.0},
        ]
    )
    res = projected_income(holdings)

    assert len(res) == 1
    assert res.iloc[0]["ticker"] == "VALID"


def test_empty_dataframe_handling():
    """
    Empty input frame yields empty output frame and zeroed summary.
    """
    empty_df = pd.DataFrame(columns=["ticker", "shares", "price", "div_yield"])
    res = projected_income(empty_df)

    assert res.empty
    assert list(res.columns) == [
        "ticker",
        "value",
        "div_yield",
        "annual_income",
        "monthly_income",
        "yield_on_cost_unknown",
    ]

    summary = income_summary(res)
    assert summary == {
        "total_value": 0.0,
        "annual_income": 0.0,
        "monthly_income": 0.0,
        "portfolio_yield": None,
        "n_payers": 0,
        "top_payers": [],
    }


def test_income_summary_mixed_payers_and_top_payers():
    """
    Test summary on mixed payers and top_payers ordering and 5-item limit.
    """
    holdings = pd.DataFrame(
        [
            {"ticker": "T1", "shares": 100, "price": 100.0, "div_yield": 1.0},  # val 10k, inc 100
            {"ticker": "T2", "shares": 100, "price": 100.0, "div_yield": 5.0},  # val 10k, inc 500
            {"ticker": "T3", "shares": 100, "price": 100.0, "div_yield": 0.0},  # val 10k, inc 0
            {"ticker": "T4", "shares": 100, "price": 100.0, "div_yield": 3.0},  # val 10k, inc 300
            {"ticker": "T5", "shares": 100, "price": 100.0, "div_yield": 4.0},  # val 10k, inc 400
            {"ticker": "T6", "shares": 100, "price": 100.0, "div_yield": 2.0},  # val 10k, inc 200
            {"ticker": "T7", "shares": 100, "price": 100.0, "div_yield": 6.0},  # val 10k, inc 600
        ]
    )
    income_df = projected_income(holdings)
    summary = income_summary(income_df)

    assert summary["total_value"] == 70000.0
    assert summary["annual_income"] == 2100.0
    assert pytest.approx(summary["monthly_income"], abs=1e-6) == 175.0
    assert pytest.approx(summary["portfolio_yield"], abs=1e-6) == 3.0
    assert summary["n_payers"] == 6

    # Top payers should be top 5 sorted descending by annual_income:
    # T7 (600), T2 (500), T5 (400), T4 (300), T6 (200)
    expected_top = [
        ("T7", 600.0),
        ("T2", 500.0),
        ("T5", 400.0),
        ("T4", 300.0),
        ("T6", 200.0),
    ]
    assert summary["top_payers"] == expected_top
