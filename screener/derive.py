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


def _relative_ratio(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns or "industry" not in df.columns or "sector" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    val = df[col].where(df[col] > 0)
    ind = df["industry"]
    sec = df["sector"]

    ind_cnt = val.groupby(ind).transform("count")
    ind_med = val.groupby(ind).transform("median")

    sec_cnt = val.groupby(sec).transform("count")
    sec_med = val.groupby(sec).transform("median")

    med = ind_med.where(ind_cnt >= 5, sec_med.where(sec_cnt >= 5))
    res = (val / med - 1) * 100.0
    return res.astype(float)


def _relative_diff(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns or "sector" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    val = df[col]
    sec = df["sector"]

    sec_cnt = val.groupby(sec).transform("count")
    sec_med = val.groupby(sec).transform("median")

    res = val - sec_med.where(sec_cnt >= 5)
    return res.astype(float)


def relative_to_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "pe_vs_industry": _relative_ratio(df, "pe"),
        "ps_vs_industry": _relative_ratio(df, "ps"),
        "sales_growth_vs_sector": _relative_diff(df, "sales_growth"),
    }


def _with(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """df with `cols` set (replacing columns of the same name, like df[k] = v), in one concat."""
    return pd.concat([df.drop(columns=list(cols), errors="ignore"), pd.DataFrame(cols, index=df.index)], axis=1)


def add_derived(raw: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    df = raw.copy()
    # Snapshots saved by an older version lack newer fields: add them as missing (NaN never passes a filter).
    from screener.fields import provider_columns
    missing = [k for k in provider_columns() if k not in df.columns]
    if missing:
        missing_dict = {k: None if COLUMNS[k].kind == "text" else np.nan for k in missing}
        if "earnings_date" in missing:
            missing_dict["earnings_date"] = pd.to_datetime(pd.Series(np.nan, index=df.index))
        df = _with(df, missing_dict)

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

    days = (pd.to_datetime(df.get("earnings_date"), errors="coerce") - pd.Timestamp(today)).dt.days
    der["days_to_earnings"] = days.where(days >= 0).astype(float)

    der.update(relative_to_groups(df))

    df = _with(df, der).replace([np.inf, -np.inf], np.nan)
    signals, n_signals = summarize(df)
    df = _with(df, {"signals": signals, "n_signals": n_signals,
                    "new": ""})     # "new" is filled by the app when a previous snapshot exists (history.py)

    return add_scores(df)
