"""
Position liquidity metrics module.

Calculates days to exit and share of average volume for stock positions,
and classifies liquidity levels and portfolio summary statistics.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LIQUIDITY_COLUMNS = ["ticker", "shares", "avg_volume", "pct_of_adv", "days_to_exit", "level"]


def liquidity_table(holdings: pd.DataFrame, participation: float = 0.10) -> pd.DataFrame:
    """Calculate liquidity metrics for position holdings.

    Parameters
    ----------
    holdings : pd.DataFrame
        DataFrame with columns 'ticker', 'shares', and 'avg_volume'.
    participation : float, default 0.10
        Max daily volume participation rate, in range (0, 1].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, shares, avg_volume, pct_of_adv, days_to_exit, level.
    """
    if not isinstance(participation, (int, float)) or np.isnan(participation) or participation <= 0 or participation > 1:
        raise ValueError(f"participation must be in (0, 1], got {participation}")

    if holdings.empty:
        return pd.DataFrame(columns=LIQUIDITY_COLUMNS)

    ticker = holdings["ticker"] if "ticker" in holdings.columns else pd.Series(dtype=object)
    shares = holdings["shares"].astype(float) if "shares" in holdings.columns else pd.Series(dtype=float)
    avg_vol = holdings["avg_volume"].astype(float) if "avg_volume" in holdings.columns else pd.Series(dtype=float)

    valid_mask = avg_vol.notna() & (avg_vol > 0) & shares.notna()

    pct_of_adv = pd.Series(np.nan, index=holdings.index, dtype=float)
    days_to_exit = pd.Series(np.nan, index=holdings.index, dtype=float)
    level = pd.Series("unknown", index=holdings.index, dtype=object)

    if valid_mask.any():
        pct_of_adv[valid_mask] = (shares[valid_mask] / avg_vol[valid_mask]) * 100.0
        days_to_exit[valid_mask] = shares[valid_mask] / (avg_vol[valid_mask] * float(participation))

        valid_days = days_to_exit[valid_mask]
        lvl_valid = pd.Series("low", index=valid_days.index, dtype=object)
        lvl_valid[valid_days <= 5.0] = "medium"
        lvl_valid[valid_days <= 1.0] = "high"

        level[valid_mask] = lvl_valid

    res = pd.DataFrame(
        {
            "ticker": ticker,
            "shares": shares,
            "avg_volume": avg_vol,
            "pct_of_adv": pct_of_adv,
            "days_to_exit": days_to_exit,
            "level": level,
        },
        columns=LIQUIDITY_COLUMNS,
    )

    return res


def summary(table: pd.DataFrame) -> dict[str, Any]:
    """Summarize liquidity risks across position liquidity table.

    Parameters
    ----------
    table : pd.DataFrame
        Liquidity table DataFrame returned by `liquidity_table`.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - n_low: count of positions with 'low' liquidity level
        - n_unknown: count of positions with 'unknown' liquidity level
        - worst: ticker with the highest days_to_exit (or None)
    """
    if table.empty or "level" not in table.columns:
        return {
            "n_low": 0,
            "n_unknown": 0,
            "worst": None,
        }

    n_low = int((table["level"] == "low").sum())
    n_unknown = int((table["level"] == "unknown").sum())

    worst = None
    if "days_to_exit" in table.columns and "ticker" in table.columns:
        valid_days = table.dropna(subset=["days_to_exit"])
        if not valid_days.empty:
            max_idx = valid_days["days_to_exit"].idxmax()
            worst = valid_days.loc[max_idx, "ticker"]

    return {
        "n_low": n_low,
        "n_unknown": n_unknown,
        "worst": worst,
    }
