"""
mc_regime.py — Market regime classification based on SPY trend + VIX level.

Defines four regimes based on historical experience:

  Bull-Calm:    SPY > SMA-200  AND  VIX < 20   (~normal bull markets)
  Bull-Anxious: SPY > SMA-200  AND  VIX ≥ 20   (rising but tense — late-cycle)
  Correction:   SPY < SMA-200  AND  VIX < 25   (orderly decline)
  Crisis:       SPY < SMA-200  AND  VIX ≥ 25   (panic — 2008, 2020, 2022)

The 25 threshold for Crisis is deliberately lower than the classical 30,
so we capture 2011, 2015, 2018 Q4, and 2022 inflation shock — not only 2008/2020.

Usage:
    from mc_regime import classify_regimes, REGIME_COLORS
    df = classify_regimes(benchmarks_df)      # adds 'Regime' column
    df['Regime'].value_counts()               # how many days in each
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("mc_regime")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# The four regimes, in display order (worst → best for color mapping)
REGIMES = ["Crisis", "Correction", "Bull-Anxious", "Bull-Calm"]

# Colors for visualization (matches dashboard dark theme)
REGIME_COLORS = {
    "Bull-Calm":    "rgba(46, 160, 67, 0.12)",     # green tint
    "Bull-Anxious": "rgba(241, 196, 15, 0.12)",    # yellow tint
    "Correction":   "rgba(230, 126, 34, 0.14)",    # orange tint
    "Crisis":       "rgba(248, 81, 73, 0.18)",     # red tint
    "Unknown":      "rgba(100, 100, 100, 0.06)",   # grey, for missing data
}

# Solid colors (for bars, markers)
REGIME_COLORS_SOLID = {
    "Bull-Calm":    "#2ea043",
    "Bull-Anxious": "#f1c40f",
    "Correction":   "#e67e22",
    "Crisis":       "#f85149",
    "Unknown":      "#6b7280",
}

# Thresholds — tuned based on 20 years of market history
VIX_BULL_CALM_MAX  = 20.0   # VIX below this = calm
VIX_CRISIS_MIN     = 25.0   # VIX above this in downtrend = crisis
SPY_SMA_WINDOW     = 200    # 200-day SMA for trend filter


# ═══════════════════════════════════════════════════════════════════════════
# Core classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_regimes(benchmarks_df: pd.DataFrame,
                       spy_col: str = "SPY_Close",
                       vix_col: str = "VIX_Close",
                       date_col: str = "Date") -> pd.DataFrame:
    """
    Add a 'Regime' column to benchmarks DataFrame based on SPY trend + VIX.

    Args:
        benchmarks_df: must contain SPY_Close, VIX_Close, Date columns
        spy_col:       name of SPY close column
        vix_col:       name of VIX close column
        date_col:      name of date column

    Returns:
        Copy of benchmarks_df with added 'Regime', 'SPY_SMA200', 'In_Uptrend' columns
    """
    if benchmarks_df is None or benchmarks_df.empty:
        return pd.DataFrame()

    df = benchmarks_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Check required columns
    missing = [c for c in [spy_col, vix_col, date_col] if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for regime classification: {missing}")
        df["Regime"] = "Unknown"
        return df

    # Compute 200-day SMA of SPY
    df["SPY_SMA200"] = df[spy_col].rolling(window=SPY_SMA_WINDOW, min_periods=100).mean()

    # Classify each day
    def _classify_row(row) -> str:
        spy = row[spy_col]
        vix = row[vix_col]
        sma = row["SPY_SMA200"]

        if pd.isna(spy) or pd.isna(vix) or pd.isna(sma):
            return "Unknown"

        in_uptrend = spy > sma
        vix_level = float(vix)

        if in_uptrend:
            if vix_level < VIX_BULL_CALM_MAX:
                return "Bull-Calm"
            else:
                return "Bull-Anxious"
        else:  # downtrend
            if vix_level < VIX_CRISIS_MIN:
                return "Correction"
            else:
                return "Crisis"

    df["In_Uptrend"] = df[spy_col] > df["SPY_SMA200"]
    df["Regime"] = df.apply(_classify_row, axis=1)

    return df


def load_and_classify(benchmarks_path: str | Path) -> pd.DataFrame:
    """Convenience: load benchmarks parquet and add regime column."""
    path = Path(benchmarks_path)
    if not path.exists():
        logger.warning(f"Benchmarks file not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return classify_regimes(df)


# ═══════════════════════════════════════════════════════════════════════════
# Regime periods — contiguous runs of the same regime
# ═══════════════════════════════════════════════════════════════════════════

def extract_regime_periods(classified_df: pd.DataFrame,
                             min_days: int = 5) -> pd.DataFrame:
    """
    Collapse day-level regime classification into periods.
    Returns DataFrame with columns: start_date, end_date, regime, n_days.

    A period is a contiguous run of the same regime lasting at least min_days
    (filters out 1-2 day flickers between regimes).
    """
    if classified_df is None or classified_df.empty or "Regime" not in classified_df.columns:
        return pd.DataFrame(columns=["start_date", "end_date", "regime", "n_days"])

    df = classified_df.sort_values("Date").reset_index(drop=True)
    # Detect regime changes
    df["regime_changed"] = (df["Regime"] != df["Regime"].shift()).astype(int)
    df["period_id"] = df["regime_changed"].cumsum()

    periods = df.groupby("period_id").agg(
        start_date=("Date", "first"),
        end_date=("Date", "last"),
        regime=("Regime", "first"),
        n_days=("Date", "count"),
    ).reset_index(drop=True)

    # Filter short flickers
    periods = periods[periods["n_days"] >= min_days].reset_index(drop=True)

    return periods


# ═══════════════════════════════════════════════════════════════════════════
# Per-regime performance attribution
# ═══════════════════════════════════════════════════════════════════════════

def compute_regime_performance(equity_df: pd.DataFrame,
                                 classified_benchmarks: pd.DataFrame,
                                 spy_capital: Optional[pd.DataFrame] = None
                                 ) -> pd.DataFrame:
    """
    Break down a single strategy's performance by market regime.

    For each regime, compute:
      - total days in regime
      - strategy return % during those days
      - annualized strategy return
      - SPY return % during those days (if spy_capital provided)
      - excess = strategy - SPY

    Args:
        equity_df:              ['Date', 'Capital'] for the strategy
        classified_benchmarks:  output of classify_regimes() — has 'Regime' column
        spy_capital:            optional ['Date', 'SPY_Capital'] for comparison

    Returns:
        DataFrame indexed by regime with performance metrics
    """
    if equity_df is None or equity_df.empty:
        return pd.DataFrame()

    eq = equity_df.copy()
    eq["Date"] = pd.to_datetime(eq["Date"]).dt.normalize()
    eq = eq.sort_values("Date").reset_index(drop=True)
    eq["daily_ret"] = eq["Capital"].pct_change()

    bench = classified_benchmarks[["Date", "Regime"]].copy()
    bench["Date"] = pd.to_datetime(bench["Date"]).dt.normalize()

    merged = eq.merge(bench, on="Date", how="inner")

    if spy_capital is not None and not spy_capital.empty:
        sp = spy_capital.copy()
        sp["Date"] = pd.to_datetime(sp["Date"]).dt.normalize()
        sp["spy_daily_ret"] = sp["SPY_Capital"].pct_change()
        merged = merged.merge(sp[["Date", "spy_daily_ret"]], on="Date", how="left")

    # Aggregate by regime
    rows = []
    for regime in REGIMES + ["Unknown"]:
        sub = merged[merged["Regime"] == regime]
        if sub.empty:
            continue

        n_days = len(sub)
        # Compound the daily returns
        strat_ret_pct = (float((1 + sub["daily_ret"].fillna(0)).prod()) - 1) * 100
        years_in_regime = n_days / 252.0
        if years_in_regime > 0 and (1 + strat_ret_pct/100) > 0:
            annualized = ((1 + strat_ret_pct/100) ** (1 / years_in_regime) - 1) * 100
        else:
            annualized = float("nan")

        row = {
            "Regime": regime,
            "Days": n_days,
            "Days_Pct": round(n_days / len(merged) * 100, 1) if len(merged) > 0 else 0,
            "Strategy_Return_Pct": round(strat_ret_pct, 2),
            "Strategy_Annualized_Pct": round(annualized, 2) if not np.isnan(annualized) else None,
        }

        if "spy_daily_ret" in sub.columns:
            spy_ret_pct = (float((1 + sub["spy_daily_ret"].fillna(0)).prod()) - 1) * 100
            if years_in_regime > 0 and (1 + spy_ret_pct/100) > 0:
                spy_annualized = ((1 + spy_ret_pct/100) ** (1 / years_in_regime) - 1) * 100
            else:
                spy_annualized = float("nan")

            row["SPY_Return_Pct"] = round(spy_ret_pct, 2)
            row["SPY_Annualized_Pct"] = round(spy_annualized, 2) if not np.isnan(spy_annualized) else None
            row["Excess_Return_Pct"] = round(strat_ret_pct - spy_ret_pct, 2)

        rows.append(row)

    result = pd.DataFrame(rows)
    # Order by canonical regime order
    order_map = {r: i for i, r in enumerate(REGIMES + ["Unknown"])}
    result["order"] = result["Regime"].map(order_map)
    result = result.sort_values("order").drop(columns="order").reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Utility: convert regime periods to Plotly shape specs for background shading
# ═══════════════════════════════════════════════════════════════════════════

def regime_shapes_for_plotly(classified_df: pd.DataFrame,
                              min_days: int = 10,
                              opacity_scale: float = 1.0) -> List[Dict]:
    """
    Convert classified benchmarks into Plotly shape dicts for background shading.

    Args:
        classified_df: output of classify_regimes()
        min_days:      minimum period length to shade (filters flickers)
        opacity_scale: scale factor for opacity (1.0 = default)

    Returns:
        List of dicts ready to pass as fig.update_layout(shapes=[...])
    """
    if classified_df is None or classified_df.empty:
        return []

    periods = extract_regime_periods(classified_df, min_days=min_days)
    shapes = []
    for _, p in periods.iterrows():
        color = REGIME_COLORS.get(p["regime"], REGIME_COLORS["Unknown"])
        # Apply opacity scaling by modifying the alpha in rgba
        if opacity_scale != 1.0 and color.startswith("rgba("):
            parts = color.replace("rgba(", "").replace(")", "").split(",")
            new_alpha = float(parts[3].strip()) * opacity_scale
            color = f"rgba({parts[0].strip()},{parts[1].strip()},{parts[2].strip()},{new_alpha:.3f})"

        shapes.append({
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": p["start_date"],
            "x1": p["end_date"],
            "y0": 0,
            "y1": 1,
            "fillcolor": color,
            "line_width": 0,
            "layer": "below",
        })

    return shapes


def regime_legend_annotations() -> str:
    """HTML snippet for a compact regime legend (for dashboard)."""
    items = []
    descriptions = {
        "Bull-Calm":    "SPY > SMA-200, VIX < 20",
        "Bull-Anxious": "SPY > SMA-200, VIX ≥ 20",
        "Correction":   "SPY < SMA-200, VIX < 25",
        "Crisis":       "SPY < SMA-200, VIX ≥ 25",
    }
    for r in REGIMES:
        color = REGIME_COLORS_SOLID[r]
        desc = descriptions[r]
        items.append(
            f'<span style="display:inline-block;padding:3px 10px;margin:2px;'
            f'background:{color};color:#111;font-size:11px;border-radius:3px;'
            f'font-weight:500;"><strong>{r}</strong> — {desc}</span>'
        )
    return '<div style="margin: 6px 0; line-height: 1.8;">' + " ".join(items) + "</div>"
