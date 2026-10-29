"""
Portfolio cash balance ledger module.

Manages cash flows including deposits, withdrawals, dividends, and interest payments.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass


@dataclass
class CashEntry:
    date: str
    kind: str  # "deposit", "withdrawal", "dividend", "interest"
    amount: float
    ticker: str | None = None
    note: str = ""


def compute_cash_balance(entries: list[CashEntry]) -> float:
    """Compute current net cash balance from a list of cash entries.

    Deposits, dividends, and interest add to balance; withdrawals subtract.
    """
    balance = 0.0
    for entry in entries:
        if entry.kind in ("deposit", "dividend", "interest"):
            balance += entry.amount
        elif entry.kind == "withdrawal":
            balance -= entry.amount
    return balance


def _parse_entry_year(date_str: str) -> int | None:
    """Extract 4-digit year from a date string."""
    try:
        return dt.date.fromisoformat(date_str).year
    except ValueError:
        m = re.search(r"\b(19|20)\d{2}\b", date_str)
        if m:
            return int(m.group(0))
        return None


def filter_dividends(entries: list[CashEntry], year: int | None = None) -> list[CashEntry]:
    """Filter dividend entries, optionally for a specific year."""
    dividends = [e for e in entries if e.kind == "dividend"]
    if year is None:
        return dividends
    return [e for e in dividends if _parse_entry_year(e.date) == year]


def summarize_cash_flows(entries: list[CashEntry]) -> dict[str, float]:
    """Summarize cash flows returning total_deposited, total_withdrawn, total_dividends, and net_cash."""
    total_deposited = sum(e.amount for e in entries if e.kind == "deposit")
    total_withdrawn = sum(e.amount for e in entries if e.kind == "withdrawal")
    total_dividends = sum(e.amount for e in entries if e.kind == "dividend")
    net_cash = compute_cash_balance(entries)

    return {
        "total_deposited": total_deposited,
        "total_withdrawn": total_withdrawn,
        "total_dividends": total_dividends,
        "net_cash": net_cash,
    }
