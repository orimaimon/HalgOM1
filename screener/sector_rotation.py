"""
Sector rotation quadrant analysis (RRG-style).

Classifies market sectors into Leading, Weakening, Lagging, and Improving relative to the market benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from screener import groups, i18n
from screener.theme import ACCENT, GAIN, LOSS, MUTED

ROTATION_COLUMNS = [
    "sector",
    "stocks",
    "market_cap",
    "perf_w",
    "perf_m",
    "perf_q",
    "rs_trend",
    "rs_momentum",
    "quadrant",
    "quadrant_he",
]

QUADRANTS_HE = {
    "Leading": "מובילים",
    "Weakening": "נחלשים",
    "Lagging": "מפגרים",
    "Improving": "משתפרים",
}

QUADRANT_COLORS = {
    "Leading": GAIN,
    "Improving": ACCENT,
    "Weakening": MUTED,
    "Lagging": LOSS,
}


def rotation_table(df: pd.DataFrame, weight: str = "median") -> pd.DataFrame:
    """Aggregates sector performance and computes relative rotation metrics (rs_trend, rs_momentum, quadrant)."""
    if df.empty or any(col not in df.columns for col in ("sector", "perf_q", "perf_m")):
        return pd.DataFrame(columns=ROTATION_COLUMNS)

    grp = groups.group_performance(df, by="sector", weight=weight)
    if grp.empty:
        return pd.DataFrame(columns=ROTATION_COLUMNS)

    grp = grp.dropna(subset=["perf_q", "perf_m"]).copy()
    if grp.empty:
        return pd.DataFrame(columns=ROTATION_COLUMNS)

    grp = grp.rename(columns={"group": "sector"})

    w = df["market_cap"].where(df["market_cap"] > 0) if "market_cap" in df.columns else None

    if weight == "cap" and w is not None and w.notna().any() and w.sum() > 0:
        ok_q = df["perf_q"].notna() & w.notna()
        bench_q = float((df["perf_q"] * w)[ok_q].sum() / w[ok_q].sum()) if ok_q.any() else np.nan

        ok_m = df["perf_m"].notna() & w.notna()
        bench_m = float((df["perf_m"] * w)[ok_m].sum() / w[ok_m].sum()) if ok_m.any() else np.nan
    elif weight == "equal":
        bench_q = float(df["perf_q"].mean())
        bench_m = float(df["perf_m"].mean())
    else:  # median
        bench_q = float(df["perf_q"].median())
        bench_m = float(df["perf_m"].median())

    if pd.isna(bench_q) or pd.isna(bench_m):
        return pd.DataFrame(columns=ROTATION_COLUMNS)

    grp["rs_trend"] = grp["perf_q"] - bench_q
    grp["rs_momentum"] = (grp["perf_m"] - bench_m) - (grp["rs_trend"] / 3.0)

    def classify(row):
        tr = row["rs_trend"]
        mo = row["rs_momentum"]
        if tr >= 0 and mo >= 0:
            return "Leading"
        elif tr >= 0 and mo < 0:
            return "Weakening"
        elif tr < 0 and mo < 0:
            return "Lagging"
        else:  # tr < 0 and mo >= 0
            return "Improving"

    grp["quadrant"] = grp.apply(classify, axis=1)
    grp["quadrant_he"] = grp["quadrant"].map(QUADRANTS_HE)

    grp = grp.sort_values("rs_trend", ascending=False).reset_index(drop=True)

    for col in ROTATION_COLUMNS:
        if col not in grp.columns:
            grp[col] = np.nan

    return grp[ROTATION_COLUMNS]


def compute_sector_rotation(
    universe: pd.DataFrame,
    snapshots_history: list | None = None,
    weight: str = "median",
) -> pd.DataFrame:
    """Computes pure sector rotation momentum leaderboard ranking sectors by performance across timeframes.

    Aggregates sector returns for available return columns (change, perf_w, perf_m, perf_q, perf_h),
    ranks sectors 1 to N per timeframe, computes a composite momentum score (average rank),
    and returns a sorted leaderboard DataFrame.
    """
    timeframe_cols = ["change", "perf_w", "perf_m", "perf_q", "perf_h"]

    if universe.empty or "sector" not in universe.columns:
        return pd.DataFrame()

    df = universe.copy()
    for col in ("market_cap", "price", "sma50"):
        if col not in df.columns:
            df[col] = np.nan

    grp = groups.group_performance(df, by="sector", weight=weight)
    if grp.empty:
        return pd.DataFrame()

    if "group" in grp.columns:
        grp = grp.rename(columns={"group": "sector"})

    avail_cols = [col for col in timeframe_cols if col in grp.columns and grp[col].notna().any()]
    if not avail_cols:
        return pd.DataFrame()

    rank_cols = []
    for col in avail_cols:
        rank_col = f"rank_{col}"
        grp[rank_col] = grp[col].rank(ascending=False, method="min").astype(int)
        rank_cols.append(rank_col)

    grp["composite_score"] = grp[rank_cols].mean(axis=1)
    grp["composite_rank"] = grp["composite_score"].rank(ascending=True, method="min").astype(int)

    grp = grp.sort_values(["composite_score"] + avail_cols[::-1], ascending=[True] + [False] * len(avail_cols)).reset_index(drop=True)

    return grp


def rotation_figure(rot_df: pd.DataFrame, lang: str = i18n.HE) -> go.Figure:
    """Builds a Plotly scatter plot (rs_trend vs rs_momentum) with reference lines and quadrant colors."""
    fig = go.Figure()

    if not rot_df.empty and "rs_trend" in rot_df.columns and "rs_momentum" in rot_df.columns:
        labels = [i18n.value("sector", str(s), lang) for s in rot_df["sector"]]
        colors = [QUADRANT_COLORS.get(q, MUTED) for q in rot_df["quadrant"]]

        fig.add_trace(
            go.Scatter(
                x=rot_df["rs_trend"],
                y=rot_df["rs_momentum"],
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=12, color=colors),
                hovertemplate="<b>%{text}</b><br>RS Trend: %{x:+.2f}%<br>RS Momentum: %{y:+.2f}%<extra></extra>",
            )
        )

    fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)

    fig.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        height=450,
        xaxis_title=i18n.ui("RS trend (3M)", lang),
        yaxis_title=i18n.ui("RS momentum (1M)", lang),
        hovermode="closest",
    )

    return fig
