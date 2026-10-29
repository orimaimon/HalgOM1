"""Monthly seasonality statistics calculation module."""

from typing import Any
import numpy as np
import pandas as pd


def _empty_seasonality_table() -> pd.DataFrame:
    """Return empty 12-row seasonality DataFrame with default values."""
    return pd.DataFrame(
        {
            "month": list(range(1, 13)),
            "avg_return": [np.nan] * 12,
            "median_return": [np.nan] * 12,
            "win_rate": [np.nan] * 12,
            "n_years": [0] * 12,
        }
    )


def monthly_seasonality(prices: pd.Series) -> pd.DataFrame:
    """Compute average return and stats by calendar month from a daily price series.

    Parameters
    ----------
    prices : pd.Series
        Daily closes indexed by DatetimeIndex (can be unsorted or contain NaNs).

    Returns
    -------
    pd.DataFrame
        12 rows (month 1..12) with columns: month, avg_return, median_return,
        win_rate, n_years.
    """
    if not isinstance(prices, pd.Series):
        return _empty_seasonality_table()

    if not isinstance(prices.index, pd.DatetimeIndex):
        return _empty_seasonality_table()

    # Clean non-numeric values and NaNs, then sort index
    cleaned = pd.to_numeric(prices, errors="coerce").dropna()
    cleaned = cleaned[cleaned.index.notna()].sort_index()

    if len(cleaned) < 2:
        return _empty_seasonality_table()

    # Resample to month-end
    m_prices = cleaned.resample("ME").last()
    if len(m_prices) < 2:
        return _empty_seasonality_table()

    # Calculate month-over-month simple return in percent (5.0 == 5%)
    m_returns = m_prices.pct_change() * 100.0

    records: list[dict[str, Any]] = []
    for m in range(1, 13):
        sub = m_returns[m_returns.index.month == m].dropna()
        if len(sub) == 0:
            records.append(
                {
                    "month": m,
                    "avg_return": np.nan,
                    "median_return": np.nan,
                    "win_rate": np.nan,
                    "n_years": 0,
                }
            )
        else:
            records.append(
                {
                    "month": m,
                    "avg_return": float(sub.mean()),
                    "median_return": float(sub.median()),
                    "win_rate": float((sub > 0).mean() * 100.0),
                    "n_years": int(len(sub)),
                }
            )

    return pd.DataFrame(records)


def best_worst(table: pd.DataFrame) -> dict[str, int | None]:
    """Find best and worst calendar months by avg_return.

    Parameters
    ----------
    table : pd.DataFrame
        Seasonality table containing 'month' and 'avg_return' columns.

    Returns
    -------
    dict
        dict with keys 'best_month' and 'worst_month' containing month integers (1..12) or None.
    """
    if (
        table is None
        or not isinstance(table, pd.DataFrame)
        or table.empty
        or "avg_return" not in table.columns
        or "month" not in table.columns
    ):
        return {"best_month": None, "worst_month": None}

    valid = table.dropna(subset=["avg_return"])
    if valid.empty:
        return {"best_month": None, "worst_month": None}

    best_idx = valid["avg_return"].idxmax()
    worst_idx = valid["avg_return"].idxmin()

    best_month = int(valid.loc[best_idx, "month"])
    worst_month = int(valid.loc[worst_idx, "month"])

    return {"best_month": best_month, "worst_month": worst_month}
