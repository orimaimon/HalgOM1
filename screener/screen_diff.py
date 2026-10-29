"""Screen diff between two universe snapshots.

Pure functions showing which tickers entered, left, or stayed in a screen
between two universe snapshots, and calculating top movers for numeric columns.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _extract_keys(df: pd.DataFrame, key: str) -> set[str]:
    """Extract normalized ticker keys from a DataFrame column."""
    if key not in df.columns:
        return set()
    res = set()
    for val in df[key]:
        if pd.isna(val):
            continue
        t = str(val).strip().upper()
        if t:
            res.add(t)
    return res


def diff(old: pd.DataFrame, new: pd.DataFrame, key: str = "ticker") -> dict[str, Any]:
    """Compute entered, left, and stayed tickers between two result sets.

    Parameters
    ----------
    old : pd.DataFrame
        Previous result set.
    new : pd.DataFrame
        Current result set.
    key : str, default "ticker"
        Key column name to compare.

    Returns
    -------
    dict
        {
            "entered": list[str],  # Tickers in new but not in old
            "left": list[str],     # Tickers in old but not in new
            "stayed": list[str],   # Tickers in both old and new
            "n_old": int,          # Count of unique valid tickers in old
            "n_new": int,          # Count of unique valid tickers in new
        }
        Lists are sorted alphabetically.
    """
    set_old = _extract_keys(old, key)
    set_new = _extract_keys(new, key)

    entered = sorted(list(set_new - set_old))
    left = sorted(list(set_old - set_new))
    stayed = sorted(list(set_old & set_new))

    return {
        "entered": entered,
        "left": left,
        "stayed": stayed,
        "n_old": len(set_old),
        "n_new": len(set_new),
    }


def _extract_series(df: pd.DataFrame, key: str, col: str) -> dict[str, float]:
    """Extract first valid numeric column value per normalized ticker key."""
    if key not in df.columns or col not in df.columns:
        return {}
    res: dict[str, float] = {}
    for k_val, c_val in zip(df[key], df[col]):
        if pd.isna(k_val):
            continue
        t = str(k_val).strip().upper()
        if not t:
            continue
        if t in res:
            continue
        if pd.isna(c_val):
            continue
        try:
            num_val = float(c_val)
            if not math.isnan(num_val):
                res[t] = num_val
        except (ValueError, TypeError):
            continue
    return res


def movers(
    old: pd.DataFrame,
    new: pd.DataFrame,
    col: str,
    key: str = "ticker",
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute change in numeric column for tickers present in both frames.

    Parameters
    ----------
    old : pd.DataFrame
        Previous result set.
    new : pd.DataFrame
        Current result set.
    col : str
        Numeric column to calculate change for (new - old).
    key : str, default "ticker"
        Key column name.
    top_n : int, default 10
        Maximum number of top movers to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["ticker", "old", "new", "change"],
        sorted by absolute change descending (and ticker ascending for ties).
        NaN/invalid pairs dropped.
    """
    cols = ["ticker", "old", "new", "change"]
    if key not in old.columns or key not in new.columns or col not in old.columns or col not in new.columns:
        return pd.DataFrame(columns=cols)

    dict_old = _extract_series(old, key, col)
    dict_new = _extract_series(new, key, col)

    common_tickers = set(dict_old.keys()) & set(dict_new.keys())
    if not common_tickers:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for t in common_tickers:
        v_old = dict_old[t]
        v_new = dict_new[t]
        chg = v_new - v_old
        rows.append({
            "ticker": t,
            "old": float(v_old),
            "new": float(v_new),
            "change": float(chg),
            "_abs_change": abs(chg),
        })

    # Sort by absolute change descending, then ticker ascending for tie-breaking
    rows.sort(key=lambda r: (-r["_abs_change"], r["ticker"]))

    if top_n > 0:
        rows = rows[:top_n]
    else:
        rows = []

    for r in rows:
        del r["_abs_change"]

    res_df = pd.DataFrame(rows, columns=cols)
    return res_df
