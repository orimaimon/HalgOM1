"""
Derived columns, computed from the canonical raw columns — independent of the data provider.
Recomputed on every load so date-relative columns (days to earnings) stay correct.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from screener.fields import COLUMNS
from screener.scores import add_scores
from screener.signals import summarize


def _pct_from(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a / b - 1) * 100, NaN where b is missing or not positive."""
    return (a / b.where(b > 0) - 1) * 100


def add_derived(raw: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    df = raw.copy()
    # Snapshots saved by an older version lack newer fields: add them as missing (NaN never passes a filter).
    missing = [k for k, c in COLUMNS.items() if c.tv and k not in df.columns]
    for k in missing:
        df[k] = None if COLUMNS[k].kind == "text" else np.nan
    if "earnings_date" in missing:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    today = today or dt.date.today()
    p = df["price"]

    df["adj_price"] = p     # today's adjusted price is the last price; adjustment only rescales past prices
    df["dollar_volume"] = df["avg_volume"] * p
    df["fwd_pe"] = p / df["eps_next_fy"].where(df["eps_next_fy"] > 0)
    for n in (20, 50, 150, 200):
        df[f"sma{n}_pct"] = _pct_from(p, df[f"sma{n}"])
    df["sma50w_pct"] = _pct_from(df["close_w"], df["sma50_w"])
    df["high52_pct"] = _pct_from(p, df["high52"])
    df["low52_pct"] = _pct_from(p, df["low52"])
    span = (df["high52"] - df["low52"]).where(lambda s: s > 0)
    df["range52"] = ((p - df["low52"]) / span).clip(0, 1)
    df["range_1m_pct"] = _pct_from(df["high_1m"], df["low_1m"])
    df["range_5d_pct"] = _pct_from(df["high_5d"], df["low_5d"])
    df["target_upside"] = _pct_from(df["target"], p)

    days = (df["earnings_date"] - pd.Timestamp(today)).dt.days
    df["days_to_earnings"] = days.where(days >= 0).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    df["signals"], df["n_signals"] = summarize(df)
    df["new"] = ""          # filled by the app when a previous snapshot exists (history.py)
    return add_scores(df)
