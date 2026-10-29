"""
Tests for portfolio cash ledger module (screener/portfolio_cash.py).
"""
import pytest

from screener.portfolio_cash import (
    CashEntry,
    compute_cash_balance,
    filter_dividends,
    summarize_cash_flows,
)


def test_compute_cash_balance_mixed_transactions():
    entries = [
        CashEntry(date="2025-01-01", kind="deposit", amount=10000.0, note="Initial deposit"),
        CashEntry(date="2025-01-15", kind="withdrawal", amount=2000.0, note="ATM withdrawal"),
        CashEntry(date="2025-02-01", kind="dividend", amount=150.5, ticker="AAPL"),
        CashEntry(date="2025-02-15", kind="interest", amount=12.25),
        CashEntry(date="2025-03-01", kind="withdrawal", amount=500.0),
    ]
    # 10000 - 2000 + 150.5 + 12.25 - 500 = 7662.75
    assert compute_cash_balance(entries) == pytest.approx(7662.75)


def test_compute_cash_balance_negative():
    entries = [
        CashEntry(date="2025-01-01", kind="deposit", amount=500.0),
        CashEntry(date="2025-01-10", kind="withdrawal", amount=1200.0),
    ]
    # 500 - 1200 = -700
    assert compute_cash_balance(entries) == pytest.approx(-700.0)


def test_compute_cash_balance_empty():
    assert compute_cash_balance([]) == 0.0


def test_filter_dividends_all_and_by_year():
    entries = [
        CashEntry(date="2024-03-15", kind="dividend", amount=50.0, ticker="MSFT"),
        CashEntry(date="2024-06-15", kind="dividend", amount=55.0, ticker="MSFT"),
        CashEntry(date="2025-01-10", kind="deposit", amount=5000.0),
        CashEntry(date="2025-03-15", kind="dividend", amount=60.0, ticker="MSFT"),
        CashEntry(date="2025-04-01", kind="interest", amount=10.0),
    ]

    all_divs = filter_dividends(entries)
    assert len(all_divs) == 3
    assert [e.amount for e in all_divs] == [50.0, 55.0, 60.0]

    divs_2024 = filter_dividends(entries, year=2024)
    assert len(divs_2024) == 2
    assert [e.amount for e in divs_2024] == [50.0, 55.0]

    divs_2025 = filter_dividends(entries, year=2025)
    assert len(divs_2025) == 1
    assert divs_2025[0].amount == 60.0

    divs_2023 = filter_dividends(entries, year=2023)
    assert divs_2023 == []


def test_summarize_cash_flows():
    entries = [
        CashEntry(date="2025-01-01", kind="deposit", amount=10000.0),
        CashEntry(date="2025-01-05", kind="deposit", amount=5000.0),
        CashEntry(date="2025-01-15", kind="withdrawal", amount=3000.0),
        CashEntry(date="2025-02-01", kind="dividend", amount=200.0, ticker="AAPL"),
        CashEntry(date="2025-02-15", kind="dividend", amount=150.0, ticker="NVDA"),
        CashEntry(date="2025-03-01", kind="interest", amount=25.0),
    ]

    summary = summarize_cash_flows(entries)

    assert summary["total_deposited"] == pytest.approx(15000.0)
    assert summary["total_withdrawn"] == pytest.approx(3000.0)
    assert summary["total_dividends"] == pytest.approx(350.0)
    # 15000 - 3000 + 350 + 25 = 12375
    assert summary["net_cash"] == pytest.approx(12375.0)
