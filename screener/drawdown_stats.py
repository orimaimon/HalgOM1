"""
Drawdown episode statistics module.

Provides pure functions to identify and analyze drawdown episodes (peak, trough, recovery)
and calculate current drawdown percentage from equity or price time series.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPISODE_COLUMNS = [
    "peak_date",
    "trough_date",
    "recovery_date",
    "depth",
    "days_to_trough",
    "days_to_recover",
]


def current_drawdown(series: pd.Series) -> float | None:
    """Calculate current drawdown percentage below the running peak at the last point.

    Returns 0.0 at a new high (or peak equal to last point).
    Returns None for fewer than 2 valid points after dropping NaNs and non-positive values.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    s = series.dropna()
    s = s[s > 0]

    if len(s) < 2:
        return None

    running_peak = float(s.cummax().iloc[-1])
    last_val = float(s.iloc[-1])

    if last_val >= running_peak or np.isclose(last_val, running_peak):
        return 0.0

    return float((last_val - running_peak) / running_peak * 100.0)


def drawdown_episodes(
    series: pd.Series,
    min_depth: float = 5.0,
    top_n: int = 5,
) -> pd.DataFrame:
    """List drawdown episodes (peak, trough, recovery) from a price or equity series.

    An episode starts at a running peak and ends when a new peak is reached
    (or at the end of the data, flagged unrecovered).

    Parameters
    ----------
    series : pd.Series
        Price or equity series indexed by date or timestamp.
    min_depth : float, default 5.0
        Minimum drawdown depth in percent (positive number, e.g. 5.0 for 5%).
        Only episodes with depth <= -min_depth are kept.
    top_n : int, default 5
        Maximum number of deepest drawdown episodes to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: peak_date, trough_date, recovery_date,
        depth, days_to_trough, days_to_recover.
        Sorted by depth ascending (deepest first).
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    s = series.dropna()
    s = s[s > 0]

    if len(s) < 2:
        return pd.DataFrame(columns=EPISODE_COLUMNS)

    values = s.values
    index = s.index
    n = len(s)

    episodes: list[dict[str, Any]] = []

    p_idx = 0
    p_val = values[0]
    tr_idx = 0
    tr_val = values[0]
    in_dd = False

    for i in range(1, n):
        v = values[i]
        if in_dd:
            if v >= p_val:
                # Episode recovered!
                depth = float((tr_val - p_val) / p_val * 100.0)
                peak_dt = index[p_idx]
                trough_dt = index[tr_idx]
                rec_dt = index[i]
                episodes.append({
                    "peak_date": peak_dt,
                    "trough_date": trough_dt,
                    "recovery_date": rec_dt,
                    "depth": depth,
                    "days_to_trough": _calc_days(trough_dt, peak_dt),
                    "days_to_recover": _calc_days(rec_dt, peak_dt),
                })
                # Reset peak to current recovery point
                p_idx = i
                p_val = v
                tr_idx = i
                tr_val = v
                in_dd = False
            else:
                if v < tr_val:
                    tr_idx = i
                    tr_val = v
        else:
            if v >= p_val:
                p_idx = i
                p_val = v
                tr_idx = i
                tr_val = v
            else:
                in_dd = True
                tr_idx = i
                tr_val = v

    # Check unrecovered final episode
    if in_dd:
        depth = float((tr_val - p_val) / p_val * 100.0)
        peak_dt = index[p_idx]
        trough_dt = index[tr_idx]
        episodes.append({
            "peak_date": peak_dt,
            "trough_date": trough_dt,
            "recovery_date": pd.NaT,
            "depth": depth,
            "days_to_trough": _calc_days(trough_dt, peak_dt),
            "days_to_recover": float("nan"),
        })

    if not episodes:
        return pd.DataFrame(columns=EPISODE_COLUMNS)

    df = pd.DataFrame(episodes)

    # Filter depth <= -min_depth
    min_d_pct = -abs(min_depth)
    df = df[df["depth"] <= min_d_pct].copy()

    if df.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS)

    # Sort by depth ascending (deepest first) and keep top_n
    df = df.sort_values("depth", ascending=True).head(top_n).reset_index(drop=True)
    return df[EPISODE_COLUMNS]


def _calc_days(d1: Any, d2: Any) -> float:
    """Calculate calendar days or index distance between d1 and d2 (d1 - d2)."""
    if pd.isna(d1) or pd.isna(d2):
        return float("nan")
    if isinstance(d1, (int, float, np.integer, np.floating)) and isinstance(d2, (int, float, np.integer, np.floating)):
        return float(d1 - d2)
    try:
        dt1 = pd.to_datetime(d1)
        dt2 = pd.to_datetime(d2)
        return float((dt1 - dt2).days)
    except Exception:
        pass
    try:
        return float(d1 - d2)
    except Exception:
        return float("nan")
