"""
Portfolio asset allocation and rebalancing module.

Calculates drift from target percentages and generates rebalancing action plans
(BUY / SELL / HOLD) with dollar amounts and suggested share quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RebalanceTarget:
    name: str
    target_pct: float


def validate_targets(target_allocations: dict[str, float]) -> bool:
    """Validates that target allocation percentages sum to 100.0% within a 0.01 tolerance."""
    if not isinstance(target_allocations, dict) or not target_allocations:
        return False
    try:
        total = sum(float(val) for val in target_allocations.values())
        return abs(total - 100.0) <= 0.0100001
    except (TypeError, ValueError):
        return False


def compute_rebalance_plan(
    current_holdings: dict[str, dict[str, Any]],
    target_allocations: dict[str, float],
    total_portfolio_value: float,
    drift_threshold_pct: float = 2.0,
) -> list[dict[str, Any]]:
    """Computes a rebalance plan for portfolio holdings given target allocations.

    Parameters:
      current_holdings: Map of ticker to holding details (e.g. {"value": 2500.0, "price": 150.0}).
      target_allocations: Map of ticker to target allocation percentage (0.0 to 100.0).
      total_portfolio_value: Total portfolio value in USD.
      drift_threshold_pct: Maximum allowed percentage drift before triggering BUY/SELL (default 2.0%).

    Returns list of dicts, each containing:
      - ticker: str
      - current_value: float
      - current_pct: float
      - target_pct: float
      - drift_pct: float
      - action: "BUY" | "SELL" | "HOLD"
      - amount_usd: float
      - suggested_shares: int
    """
    tickers = sorted(set(current_holdings.keys()) | set(target_allocations.keys()))
    plan: list[dict[str, Any]] = []

    for ticker in tickers:
        holding = current_holdings.get(ticker, {})
        if isinstance(holding, dict):
            if "value" in holding and holding["value"] is not None:
                curr_val = float(holding["value"])
            elif "current_value" in holding and holding["current_value"] is not None:
                curr_val = float(holding["current_value"])
            elif "shares" in holding and "price" in holding and holding["shares"] is not None and holding["price"] is not None:
                curr_val = float(holding["shares"]) * float(holding["price"])
            elif "quantity" in holding and "price" in holding and holding["quantity"] is not None and holding["price"] is not None:
                curr_val = float(holding["quantity"]) * float(holding["price"])
            else:
                curr_val = 0.0

            price = float(holding.get("price", holding.get("last_price", 0.0)) or 0.0)
        else:
            curr_val = 0.0
            price = 0.0

        if total_portfolio_value > 0:
            current_pct = round((curr_val / total_portfolio_value) * 100.0, 2)
        else:
            current_pct = 0.0

        target_pct = round(float(target_allocations.get(ticker, 0.0)), 2)
        drift_pct = round(current_pct - target_pct, 2)

        if drift_pct > drift_threshold_pct:
            action = "SELL"
        elif drift_pct < -drift_threshold_pct:
            action = "BUY"
        else:
            action = "HOLD"

        target_val = total_portfolio_value * (target_pct / 100.0)
        diff_usd = target_val - curr_val

        if action == "HOLD":
            amount_usd = 0.0
            suggested_shares = 0
        else:
            amount_usd = round(abs(diff_usd), 2)
            if price > 0:
                suggested_shares = int(amount_usd / price)
            else:
                suggested_shares = 0

        plan.append({
            "ticker": ticker,
            "current_value": curr_val,
            "current_pct": current_pct,
            "target_pct": target_pct,
            "drift_pct": drift_pct,
            "action": action,
            "amount_usd": amount_usd,
            "suggested_shares": suggested_shares,
        })

    return plan
