import datetime as dt
import pandas as pd
import pytest

from screener.portfolio_tax import (
    TAX_DISCLAIMER_HE,
    compute_israeli_capital_gains,
    summarize_tax_year,
)


def test_tax_disclaimer_constant():
    assert TAX_DISCLAIMER_HE == "החישוב נועד לאומדן כללי בלבד ואינו מהווה ייעוץ מס או תחליף לדוח רשמי."


def test_compute_israeli_capital_gains_empty():
    df = pd.DataFrame()
    res = compute_israeli_capital_gains(df)
    assert res.empty
    for col in ["cost_basis_ils", "proceeds_ils", "nominal_gain_ils", "estimated_tax_ils"]:
        assert col in res.columns


def test_compute_israeli_capital_gains_positive_gains_with_rates():
    # Buy $1000 on 2024-01-01 (rate 3.6), sell $1500 on 2024-06-01 (rate 3.7)
    trades = [
        {
            "ticker": "AAPL",
            "date": dt.date(2024, 6, 1),
            "buy_date": dt.date(2024, 1, 1),
            "cost_basis": 1000.0,
            "proceeds": 1500.0,
        }
    ]
    closed_df = pd.DataFrame(trades)
    rates = {
        "2024-01-01": 3.6,
        "2024-06-01": 3.7,
    }

    res = compute_israeli_capital_gains(closed_df, usd_ils_rates=rates)

    # Cost basis ILS: 1000 * 3.6 = 3600
    # Proceeds ILS: 1500 * 3.7 = 5550
    # Nominal gain ILS: 5550 - 3600 = 1950
    # Tax ILS: 1950 * 0.25 = 487.5
    assert res.loc[0, "cost_basis_ils"] == pytest.approx(3600.0)
    assert res.loc[0, "proceeds_ils"] == pytest.approx(5550.0)
    assert res.loc[0, "nominal_gain_ils"] == pytest.approx(1950.0)
    assert res.loc[0, "estimated_tax_ils"] == pytest.approx(487.5)


def test_compute_israeli_capital_gains_fallback_constant_rates():
    # When no rates dictionary is provided, default rate 3.6 applies
    trades = [
        {
            "ticker": "MSFT",
            "date": "2024-05-10",
            "hold_days": 10,
            "cost_basis": 2000.0,
            "proceeds": 1800.0,
        }
    ]
    closed_df = pd.DataFrame(trades)

    res = compute_israeli_capital_gains(closed_df, usd_ils_rates=None)

    # Cost basis ILS: 2000 * 3.6 = 7200
    # Proceeds ILS: 1800 * 3.6 = 6480
    # Nominal gain ILS: 6480 - 7200 = -720
    # Tax ILS: max(0, -720 * 0.25) = 0.0
    assert res.loc[0, "cost_basis_ils"] == pytest.approx(7200.0)
    assert res.loc[0, "proceeds_ils"] == pytest.approx(6480.0)
    assert res.loc[0, "nominal_gain_ils"] == pytest.approx(-720.0)
    assert res.loc[0, "estimated_tax_ils"] == pytest.approx(0.0)


def test_summarize_tax_year_loss_offsetting_and_filtering():
    trades = [
        # 2024 Trade 1: Gain +2000 ILS
        {
            "date": dt.date(2024, 3, 15),
            "cost_basis_ils": 3000.0,
            "proceeds_ils": 5000.0,
            "nominal_gain_ils": 2000.0,
        },
        # 2024 Trade 2: Loss -500 ILS
        {
            "date": dt.date(2024, 8, 20),
            "cost_basis_ils": 2000.0,
            "proceeds_ils": 1500.0,
            "nominal_gain_ils": -500.0,
        },
        # 2023 Trade (should be excluded from 2024 summary)
        {
            "date": dt.date(2023, 11, 10),
            "cost_basis_ils": 1000.0,
            "proceeds_ils": 3000.0,
            "nominal_gain_ils": 2000.0,
        },
    ]
    tax_df = pd.DataFrame(trades)

    summary_2024 = summarize_tax_year(tax_df, 2024)

    # 2024 Proceeds: 5000 + 1500 = 6500
    # Total gain: 2000
    # Total loss: 500
    # Net gain: 1500
    # Estimated tax: 1500 * 0.25 = 375
    assert summary_2024["total_proceeds_ils"] == pytest.approx(6500.0)
    assert summary_2024["total_gain_ils"] == pytest.approx(2000.0)
    assert summary_2024["total_loss_ils"] == pytest.approx(500.0)
    assert summary_2024["net_gain_ils"] == pytest.approx(1500.0)
    assert summary_2024["estimated_tax_ils"] == pytest.approx(375.0)


def test_summarize_tax_year_net_loss():
    trades = [
        {
            "date": dt.date(2024, 4, 1),
            "proceeds_ils": 1000.0,
            "nominal_gain_ils": -1500.0,
        }
    ]
    tax_df = pd.DataFrame(trades)

    summary = summarize_tax_year(tax_df, 2024)

    assert summary["total_proceeds_ils"] == pytest.approx(1000.0)
    assert summary["total_gain_ils"] == pytest.approx(0.0)
    assert summary["total_loss_ils"] == pytest.approx(1500.0)
    assert summary["net_gain_ils"] == pytest.approx(-1500.0)
    assert summary["estimated_tax_ils"] == pytest.approx(0.0)


def test_summarize_tax_year_empty():
    summary = summarize_tax_year(pd.DataFrame(), 2024)
    assert summary == {
        "total_proceeds_ils": 0.0,
        "total_gain_ils": 0.0,
        "total_loss_ils": 0.0,
        "net_gain_ils": 0.0,
        "estimated_tax_ils": 0.0,
    }
