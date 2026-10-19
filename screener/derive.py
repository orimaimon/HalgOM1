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
    from screener.fields import provider_columns
    missing = [k for k in provider_columns() if k not in df.columns]

    new_cols = {}
    for k in missing:
        new_cols[k] = None if COLUMNS[k].kind == "text" else np.nan

    # We must add these to df temporarily so they can be accessed below if needed
    if missing:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        if "earnings_date" in missing:
            df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    today = today or dt.date.today()
    p = df["price"]

    der = {}
    der["adj_price"] = p     # today's adjusted price is the last price; adjustment only rescales past prices
    der["dollar_volume"] = df["avg_volume"] * p
    der["fwd_pe"] = p / df["eps_next_fy"].where(df["eps_next_fy"] > 0)
    for n in (20, 50, 150, 200):
        der[f"sma{n}_pct"] = _pct_from(p, df[f"sma{n}"])
    der["sma50w_pct"] = _pct_from(df["close_w"], df["sma50_w"])
    der["high52_pct"] = _pct_from(p, df["high52"])
    der["low52_pct"] = _pct_from(p, df["low52"])
    span = (df["high52"] - df["low52"]).where(lambda s: s > 0)
    der["range52"] = ((p - df["low52"]) / span).clip(0, 1)
    der["range_1m_pct"] = _pct_from(df["high_1m"], df["low_1m"])
    der["range_5d_pct"] = _pct_from(df["high_5d"], df["low_5d"])
    der["target_upside"] = _pct_from(df["target"], p)

    days = (df["earnings_date"] - pd.Timestamp(today)).dt.days
    der["days_to_earnings"] = days.where(days >= 0).astype(float)

    df = pd.concat([df, pd.DataFrame(der, index=df.index)], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)

    signals, n_signals = summarize(df)

    # We need to add signals, n_signals, and new.
    # It's better to add them via concat or assigning. We can just assign them directly or add via concat.
    # Assigning 3 columns at the end usually does not trigger a fragmentation warning if done cleanly.
    # However we can also use pd.concat again for safety.
    final_cols = {
        "signals": signals,
        "n_signals": n_signals,
        "new": ""          # filled by the app when a previous snapshot exists (history.py)
    }
    df = pd.concat([df, pd.DataFrame(final_cols, index=df.index)], axis=1)

    return add_scores(df)
