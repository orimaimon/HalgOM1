"""
Tests for portfolio asset allocation and rebalancing calculation module.
"""
import pytest

from screener.portfolio_rebalance import (
    RebalanceTarget,
    compute_rebalance_plan,
    validate_targets,
)


def test_rebalance_target_dataclass():
    target = RebalanceTarget(name="AAPL", target_pct=25.0)
    assert target.name == "AAPL"
    assert target.target_pct == 25.0


def test_validate_targets_exact_100():
    targets = {"AAPL": 40.0, "MSFT": 30.0, "GOOGL": 30.0}
    assert validate_targets(targets) is True


def test_validate_targets_within_tolerance():
    targets = {"A": 33.33, "B": 33.33, "C": 33.34}
    assert validate_targets(targets) is True

    targets_low = {"A": 33.33, "B": 33.33, "C": 33.33}  # sum 99.99
    assert validate_targets(targets_low) is True

    targets_high = {"A": 33.33, "B": 33.34, "C": 33.34}  # sum 100.01
    assert validate_targets(targets_high) is True


def test_validate_targets_invalid():
    assert validate_targets({"AAPL": 50.0, "MSFT": 40.0}) is False
    assert validate_targets({"AAPL": 60.0, "MSFT": 50.0}) is False
    assert validate_targets({}) is False
    assert validate_targets(None) is False


def test_overweighted_position_triggers_sell():
    total_val = 10000.0
    targets = {"AAPL": 20.0, "MSFT": 80.0}
    holdings = {
        "AAPL": {"value": 3500.0, "price": 150.0},  # 35% current vs 20% target -> +15% drift -> SELL
        "MSFT": {"value": 6500.0, "price": 250.0},  # 65% current vs 80% target -> -15% drift -> BUY
    }

    plan = compute_rebalance_plan(
        current_holdings=holdings,
        target_allocations=targets,
        total_portfolio_value=total_val,
        drift_threshold_pct=2.0,
    )

    by_ticker = {row["ticker"]: row for row in plan}
    aapl = by_ticker["AAPL"]

    assert aapl["current_value"] == 3500.0
    assert aapl["current_pct"] == 35.0
    assert aapl["target_pct"] == 20.0
    assert aapl["drift_pct"] == 15.0
    assert aapl["action"] == "SELL"
    assert aapl["amount_usd"] == 1500.0  # target is $2000, current $3500 -> sell $1500
    assert aapl["suggested_shares"] == 10  # 1500 / 150 = 10 shares


def test_underweighted_position_triggers_buy():
    total_val = 10000.0
    targets = {"MSFT": 50.0, "GOOGL": 50.0}
    holdings = {
        "MSFT": {"value": 2000.0, "price": 200.0},  # 20% current vs 50% target -> -30% drift -> BUY
        "GOOGL": {"value": 8000.0, "price": 100.0},
    }

    plan = compute_rebalance_plan(
        current_holdings=holdings,
        target_allocations=targets,
        total_portfolio_value=total_val,
        drift_threshold_pct=2.0,
    )

    by_ticker = {row["ticker"]: row for row in plan}
    msft = by_ticker["MSFT"]

    assert msft["current_value"] == 2000.0
    assert msft["current_pct"] == 20.0
    assert msft["target_pct"] == 50.0
    assert msft["drift_pct"] == -30.0
    assert msft["action"] == "BUY"
    assert msft["amount_usd"] == 3000.0  # target is $5000, current $2000 -> buy $3000
    assert msft["suggested_shares"] == 15  # 3000 / 200 = 15 shares


def test_position_within_drift_threshold_shows_hold():
    total_val = 10000.0
    targets = {"AAPL": 30.0, "MSFT": 70.0}
    holdings = {
        "AAPL": {"value": 3100.0, "price": 150.0},  # 31% vs 30% target -> +1.0% drift <= 2.0% -> HOLD
        "MSFT": {"value": 6900.0, "price": 200.0},  # 69% vs 70% target -> -1.0% drift >= -2.0% -> HOLD
    }

    plan = compute_rebalance_plan(
        current_holdings=holdings,
        target_allocations=targets,
        total_portfolio_value=total_val,
        drift_threshold_pct=2.0,
    )

    by_ticker = {row["ticker"]: row for row in plan}

    aapl = by_ticker["AAPL"]
    assert aapl["current_pct"] == 31.0
    assert aapl["target_pct"] == 30.0
    assert aapl["drift_pct"] == 1.0
    assert aapl["action"] == "HOLD"
    assert aapl["amount_usd"] == 0.0
    assert aapl["suggested_shares"] == 0

    msft = by_ticker["MSFT"]
    assert msft["current_pct"] == 69.0
    assert msft["target_pct"] == 70.0
    assert msft["drift_pct"] == -1.0
    assert msft["action"] == "HOLD"
    assert msft["amount_usd"] == 0.0
    assert msft["suggested_shares"] == 0


def test_new_position_not_in_current_holdings():
    total_val = 10000.0
    targets = {"AAPL": 50.0, "NVDA": 50.0}
    holdings = {
        "AAPL": {"value": 10000.0, "price": 100.0},
    }

    plan = compute_rebalance_plan(
        current_holdings=holdings,
        target_allocations=targets,
        total_portfolio_value=total_val,
        drift_threshold_pct=2.0,
    )

    by_ticker = {row["ticker"]: row for row in plan}
    assert "NVDA" in by_ticker
    nvda = by_ticker["NVDA"]

    assert nvda["current_value"] == 0.0
    assert nvda["current_pct"] == 0.0
    assert nvda["target_pct"] == 50.0
    assert nvda["drift_pct"] == -50.0
    assert nvda["action"] == "BUY"
    assert nvda["amount_usd"] == 5000.0
    assert nvda["suggested_shares"] == 0  # No price provided in holdings


def test_removed_position_not_in_targets():
    total_val = 10000.0
    targets = {"AAPL": 100.0}
    holdings = {
        "AAPL": {"value": 8000.0, "price": 100.0},
        "TSLA": {"value": 2000.0, "price": 200.0},  # 20% vs 0% target -> +20% drift -> SELL
    }

    plan = compute_rebalance_plan(
        current_holdings=holdings,
        target_allocations=targets,
        total_portfolio_value=total_val,
        drift_threshold_pct=2.0,
    )

    by_ticker = {row["ticker"]: row for row in plan}
    assert "TSLA" in by_ticker
    tsla = by_ticker["TSLA"]

    assert tsla["current_value"] == 2000.0
    assert tsla["current_pct"] == 20.0
    assert tsla["target_pct"] == 0.0
    assert tsla["drift_pct"] == 20.0
    assert tsla["action"] == "SELL"
    assert tsla["amount_usd"] == 2000.0
    assert tsla["suggested_shares"] == 10
