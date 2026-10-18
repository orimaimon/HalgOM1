"""
Derived columns, computed from the canonical raw columns — independent of the data provider.
Recomputed on every load so date-relative columns (days to earnings) stay correct.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from screener.scores import add_scores


def _pct_from(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a / b - 1) * 100, NaN where b is missing or not positive."""
    return (a / b.where(b > 0) - 1) * 100


def add_derived(raw: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    df = raw.copy()
    today = today or dt.date.today()
    p = df["price"]

    df["dollar_volume"] = df["avg_volume"] * p
    df["fwd_pe"] = p / df["eps_next_fy"].where(df["eps_next_fy"] > 0)
    for n in (20, 50, 200):
        df[f"sma{n}_pct"] = _pct_from(p, df[f"sma{n}"])
    df["high52_pct"] = _pct_from(p, df["high52"])
    df["low52_pct"] = _pct_from(p, df["low52"])
    span = (df["high52"] - df["low52"]).where(lambda s: s > 0)
    df["range52"] = ((p - df["low52"]) / span).clip(0, 1)
    df["target_upside"] = _pct_from(df["target"], p)

    days = (df["earnings_date"] - pd.Timestamp(today)).dt.days
    df["days_to_earnings"] = days.where(days >= 0).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    return add_scores(df)
