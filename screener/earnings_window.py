"""
Earnings window calendar logic.

Provides pure functions to list upcoming earnings reports from the universe
and group tickers into time-horizon buckets.
"""
from __future__ import annotations

import pandas as pd


def upcoming(
    universe: pd.DataFrame,
    days: int = 14,
    held: set[str] | None = None,
) -> pd.DataFrame:
    """Returns upcoming earnings reports within `days` days from the universe.

    Parameters
    ----------
    universe : pd.DataFrame
        Stock universe DataFrame containing 'ticker' and 'days_to_earnings' columns,
        and optionally 'market_cap' and 'sector'.
    days : int, default 14
        Maximum number of days until earnings report (must be >= 0).
    held : set[str] | None, default None
        Set of tickers currently held in a portfolio (case-insensitive).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'ticker', 'days_to_earnings', 'held',
        plus 'market_cap' and 'sector' if present in universe.
    """
    if days < 0:
        raise ValueError(f"days must be >= 0, got {days}")

    cols = ["ticker", "days_to_earnings", "held"]
    if "market_cap" in universe.columns:
        cols.append("market_cap")
    if "sector" in universe.columns:
        cols.append("sector")

    if (
        universe.empty
        or "ticker" not in universe.columns
        or "days_to_earnings" not in universe.columns
    ):
        return pd.DataFrame(columns=cols)

    df = universe.copy()

    s_days = pd.to_numeric(df["days_to_earnings"], errors="coerce")
    mask = (s_days >= 0) & (s_days <= days)

    filtered = df[mask].copy()
    if filtered.empty:
        return pd.DataFrame(columns=cols)

    filtered["days_to_earnings"] = pd.to_numeric(filtered["days_to_earnings"], errors="coerce")

    held_set = {t.upper() for t in held} if held else set()
    filtered["held"] = filtered["ticker"].astype(str).str.upper().isin(held_set)

    out = filtered[cols].sort_values(
        by=["days_to_earnings", "ticker"], ascending=[True, True]
    ).reset_index(drop=True)

    return out


def bucket(df: pd.DataFrame) -> dict[str, list[str]]:
    """Groups tickers into earnings time buckets: 'today', 'this_week', 'next_week', 'later'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'ticker' and 'days_to_earnings' columns.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys 'today', 'this_week', 'next_week', 'later',
        each containing a list of ticker symbols.
    """
    res: dict[str, list[str]] = {
        "today": [],
        "this_week": [],
        "next_week": [],
        "later": [],
    }

    if (
        df is None
        or df.empty
        or "ticker" not in df.columns
        or "days_to_earnings" not in df.columns
    ):
        return res

    s_days = pd.to_numeric(df["days_to_earnings"], errors="coerce")

    for ticker_raw, d in zip(df["ticker"], s_days):
        if pd.isna(d) or d < 0:
            continue
        ticker = str(ticker_raw)
        if d == 0:
            res["today"].append(ticker)
        elif 1 <= d <= 7:
            res["this_week"].append(ticker)
        elif 8 <= d <= 14:
            res["next_week"].append(ticker)
        elif d >= 15:
            res["later"].append(ticker)

    return res
