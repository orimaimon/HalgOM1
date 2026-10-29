"""
IBD-style Relative Strength (RS) Rating.

Computes a weighted multi-period performance score across 1-quarter (perf_q: 0.4),
6-month (perf_h: 0.2), 1-year (perf_y: 0.2), and 1-month (perf_m: 0.2) returns.
The score is then converted to a 1–99 percentile rank relative to the universe.

Functions are pure with no network, disk I/O, or Streamlit dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {
    "perf_q": 0.4,
    "perf_h": 0.2,
    "perf_y": 0.2,
    "perf_m": 0.2,
}


def rs_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the weighted sum of performance columns.

    Weights: perf_q (0.4), perf_h (0.2), perf_y (0.2), perf_m (0.2).
    If a column is missing from df or a value is NaN for a row, weights are
    renormalised over the available columns for that row.
    Rows with no available columns receive np.nan.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    weight_sum = pd.Series(0.0, index=df.index, dtype=float)

    for col, w in WEIGHTS.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            valid = s.notna()
            weighted_sum = weighted_sum + s.fillna(0.0) * w * valid
            weight_sum = weight_sum + w * valid

    score = weighted_sum / weight_sum.replace(0.0, np.nan)
    return score


def rs_rating(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the IBD-style 1–99 percentile rank of rs_score over universe rows.

    Scaled to integers 1..99 (best = 99).
    NaN scores stay NaN.
    Ties share the average rank.

    Single-row behavior:
    - If the single row has a valid rs_score, its percentile rank is 1.0 (top rank),
      yielding an RS rating of 99.
    - If the single row has a NaN rs_score, its RS rating is NaN.
    """
    score = rs_score(df) if isinstance(df, pd.DataFrame) else df
    if score.empty or score.notna().sum() == 0:
        return pd.Series(np.nan, index=score.index, dtype=float)

    pct = score.rank(pct=True, na_option="keep")
    rating = (pct * 99).round().clip(1, 99)
    return rating


def add_rs_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with an 'rs_rating' column added/updated.

    Does not mutate the input DataFrame.
    """
    rating = rs_rating(df)
    out = df.copy()
    out["rs_rating"] = rating
    return out
