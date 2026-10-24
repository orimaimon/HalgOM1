"""
Portfolio correlation, high-correlation pair warnings, and market stress scenarios.

Pure calculation module:
  - return_correlations: computes wide return correlation matrix from stored daily price histories.
  - high_correlation_pairs: identifies position pairs above a correlation threshold with combined weight warnings.
  - stress_scenarios: beta-based estimated portfolio performance under market movement scenarios.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from screener.data import prices
from screener.portfolio import PList
from screener.portfolio_risk import exposure

STRESS_COLUMNS = ["market_move_pct", "est_portfolio_pct", "est_pnl_dollars", "est_value_dollars", "description_he"]


def return_correlations(tickers: Iterable[str], days: int = 90, min_overlap: int = 20) -> pd.DataFrame:
    """Return wide correlation matrix of daily percentage returns over the last `days` trading days.

    - Deduplicates tickers preserving order.
    - Reads prices.history(ticker). Prefers 'adj_close' if present and non-empty, else 'close'.
    - Normalizes index to naive dates: pd.to_datetime(idx).tz_localize(None).normalize().
    - Deduplicates dates keeping last, and sorts index.
    - Combines into a wide closes DataFrame, takes last days + 1 rows, computes daily pct_change(fill_method=None).
    - Returns .corr(min_periods=min_overlap).
    - Returns an empty DataFrame if fewer than 2 tickers have sufficient history.
    """
    unique_tickers = list(dict.fromkeys(tickers))
    if len(unique_tickers) < 2:
        return pd.DataFrame()

    closes_dict: dict[str, pd.Series] = {}
    for ticker in unique_tickers:
        hist = prices.history(ticker)
        if hist.empty:
            continue

        col = "adj_close" if "adj_close" in hist.columns and hist["adj_close"].notna().any() else "close"
        if col not in hist.columns or hist[col].isna().all():
            continue

        s = hist[col].dropna()
        if s.empty:
            continue

        # Normalize index to naive dates, deduplicate keeping last, sort
        idx = pd.to_datetime(s.index).tz_localize(None).normalize()
        s = s.copy()
        s.index = idx
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes_dict[ticker] = s

    if len(closes_dict) < 2:
        return pd.DataFrame()

    closes_df = pd.DataFrame(closes_dict)
    # Take the last days + 1 rows
    closes_df = closes_df.tail(days + 1)

    returns_df = closes_df.pct_change(fill_method=None)
    corr_df = returns_df.corr(min_periods=min_overlap)

    # Filter out tickers that resulted in all NaN in correlation matrix (e.g. insufficient overlap)
    valid_cols = [c for c in corr_df.columns if corr_df[c].notna().any()]
    if len(valid_cols) < 2:
        return pd.DataFrame()

    return corr_df


def high_correlation_pairs(
    plist: PList,
    universe: pd.DataFrame,
    threshold: float = 0.80,
    days: int = 90,
    min_overlap: int = 20,
) -> list[dict]:
    """Find position pairs with return correlation >= threshold.

    Returns list of dicts:
      - ticker_a: str
      - ticker_b: str
      - correlation: float
      - combined_weight: float
      - warning: Hebrew string
    Sorted by combined_weight descending, then correlation descending.
    """
    exp = exposure(plist, universe)
    if len(exp) < 2:
        return []

    tickers = exp["ticker"].tolist()
    weights = dict(zip(exp["ticker"], exp["weight"]))

    corr = return_correlations(tickers, days=days, min_overlap=min_overlap)
    if corr.empty:
        return []

    pairs: list[dict] = []
    n = len(tickers)
    for i in range(n):
        t1 = tickers[i]
        if t1 not in corr.columns:
            continue
        for j in range(i + 1, n):
            t2 = tickers[j]
            if t2 not in corr.columns:
                continue

            c_val = corr.loc[t1, t2]
            if pd.isna(c_val):
                continue

            c = float(c_val)
            if c >= threshold:
                w1 = float(weights.get(t1, 0.0))
                w2 = float(weights.get(t2, 0.0))
                comb_w = w1 + w2
                corr_pct = round(c * 100)
                comb_w_pct = round(comb_w)
                warning_text = f"{t1} ו-{t2} זזות יחד (מתאם {corr_pct}%) ומהוות יחד {comb_w_pct}% מהתיק"

                pairs.append({
                    "ticker_a": t1,
                    "ticker_b": t2,
                    "correlation": c,
                    "combined_weight": comb_w,
                    "warning": warning_text,
                })

    pairs.sort(key=lambda x: (x["combined_weight"], x["correlation"]), reverse=True)
    return pairs


def stress_scenarios(
    exp: pd.DataFrame,
    market_moves_pct: Iterable[float] = (-5.0, -10.0, -20.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Market stress scenarios based on weighted portfolio beta.

    Columns: ["market_move_pct", "est_portfolio_pct", "est_pnl_dollars", "est_value_dollars", "description_he"]
    """
    if exp.empty or "value" not in exp.columns or exp["value"].sum() <= 0:
        return pd.DataFrame(columns=STRESS_COLUMNS)

    total_value = float(exp["value"].sum())

    betas = exp["beta"].copy() if "beta" in exp.columns else pd.Series(1.0, index=exp.index)
    betas_filled = pd.to_numeric(betas, errors="coerce").fillna(1.0)

    eff_beta = float((exp["value"] * betas_filled).sum() / total_value)

    rows = []
    for m in market_moves_pct:
        m_float = float(m)
        est_portfolio_pct = m_float * eff_beta
        est_pnl_dollars = total_value * (est_portfolio_pct / 100.0)
        est_value_dollars = max(0.0, total_value + est_pnl_dollars)

        direction = "ירידת" if m_float < 0 else "עליית"
        abs_m = abs(m_float)
        description_he = f"ב{direction} שוק של {abs_m:.0f}%, התיק צפוי לנוע בכ-{est_portfolio_pct:+.1f}% (${est_pnl_dollars:+,.0f})"

        rows.append({
            "market_move_pct": m_float,
            "est_portfolio_pct": est_portfolio_pct,
            "est_pnl_dollars": est_pnl_dollars,
            "est_value_dollars": est_value_dollars,
            "description_he": description_he,
        })

    return pd.DataFrame(rows, columns=STRESS_COLUMNS)
