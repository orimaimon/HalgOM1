"""
Percentile scores (0–100, 100 = best in the current universe).

  Value     cheap on P/E, P/S, P/B, EV/EBITDA, P/FCF  (negative multiples = losses → worst)
  Quality   high ROE, ROIC, gross & operating margin, low Debt/Eq (negative equity → worst)
  Momentum  strong 3/6/12-month returns, near the 52W high, above SMA200
  Score     mean of the sub-scores that are available (needs at least 2)

A sub-score needs at least half of its metrics; otherwise it is NaN rather than a guess.
These are cross-sectional ranks for today's universe — not a backtested signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (column, higher_is_better, non_positive_is_worst)
VALUE = [("pe", False, True), ("ps", False, True), ("pb", False, True),
         ("ev_ebitda", False, True), ("pfcf", False, True)]
QUALITY = [("roe", True, False), ("roic", True, False), ("gross_margin", True, False),
           ("oper_margin", True, False), ("debt_eq", False, True)]
MOMENTUM = [("perf_q", True, False), ("perf_h", True, False), ("perf_y", True, False),
            ("high52_pct", True, False), ("sma200_pct", True, False)]


def _rank(s: pd.Series, higher_better: bool, non_positive_worst: bool) -> pd.Series:
    x = s.astype(float)
    if non_positive_worst:
        x = x.where(~(x <= 0), np.inf if not higher_better else -np.inf)
    if not higher_better:
        x = -x
    return x.rank(pct=True, na_option="keep") * 100


def _subscore(df: pd.DataFrame, spec) -> pd.Series:
    parts = [_rank(df[c], hb, npw) for c, hb, npw in spec if c in df.columns]
    if not parts:
        return pd.Series(np.nan, index=df.index)
    ranks = pd.concat(parts, axis=1)
    need = (len(spec) + 1) // 2
    return ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= need)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_value"] = _subscore(df, VALUE)
    df["score_quality"] = _subscore(df, QUALITY)
    df["score_momentum"] = _subscore(df, MOMENTUM)
    subs = df[["score_value", "score_quality", "score_momentum"]]
    df["score"] = subs.mean(axis=1).where(subs.notna().sum(axis=1) >= 2)
    for c in ("score", "score_value", "score_quality", "score_momentum"):
        df[c] = df[c].round(0)
    return df
