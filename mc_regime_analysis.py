"""
mc_regime_analysis.py — Per-regime performance analysis for MC dashboards.

Produces charts + tables that decompose strategy performance by market regime:
  1. Regime Timeline         — shows when each regime was active over 20 years
  2. Per-Regime Bar Chart    — avg CAGR of top 25 runs within each regime
  3. Regime Performance Table — Top 25 runs × 4 regimes matrix of returns
  4. Regime Duration Stats   — how much time was spent in each regime

All functions return HTML div strings ready for dashboard embedding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot

from mc_database import MCDatabase
from mc_regime import (
    REGIMES, REGIME_COLORS, REGIME_COLORS_SOLID,
    classify_regimes, extract_regime_periods, compute_regime_performance,
    regime_legend_annotations,
)
from mc_period_analysis import load_top_runs_with_curves, load_spy_curve

logger = logging.getLogger("mc_regime_analysis")

BG        = "#0f1419"
CARD      = "#1a2332"
TEXT      = "#e6edf3"
DIM       = "#8b949e"
POS       = "#2ea043"
NEG       = "#f85149"


def _fig_to_div(fig: go.Figure) -> str:
    return plotly_plot(fig, include_plotlyjs=False, output_type="div",
                        config={"responsive": True, "displaylogo": False})


def _empty_div(msg: str) -> str:
    return f'<div class="empty-state">{msg}</div>'


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1: Regime Timeline (horizontal bars showing each period)
# ═══════════════════════════════════════════════════════════════════════════

def build_regime_timeline(classified_df: pd.DataFrame,
                            min_days: int = 15) -> go.Figure:
    """
    Timeline showing regime history over the strategy's period.
    Each horizontal bar represents one regime period, colored by type.
    """
    if classified_df is None or classified_df.empty:
        return go.Figure()

    periods = extract_regime_periods(classified_df, min_days=min_days)
    if periods.empty:
        return go.Figure()

    fig = go.Figure()
    for regime in REGIMES:
        sub = periods[periods["regime"] == regime]
        if sub.empty:
            continue
        # Render each period as a horizontal segment
        for _, p in sub.iterrows():
            fig.add_trace(go.Scatter(
                x=[p["start_date"], p["end_date"]],
                y=[regime, regime],
                mode="lines",
                line=dict(color=REGIME_COLORS_SOLID[regime], width=22),
                name=regime,
                legendgroup=regime,
                showlegend=False,
                hovertemplate=(
                    f"<b>{regime}</b><br>"
                    f"{p['start_date'].date()} → {p['end_date'].date()}<br>"
                    f"{p['n_days']} trading days<extra></extra>"
                ),
            ))

    # Add legend entries (one per regime)
    for regime in REGIMES:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=REGIME_COLORS_SOLID[regime], size=12, symbol="square"),
            name=regime, legendgroup=regime, showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="Regime Timeline — 20 Years of Market State",
                   font=dict(size=15, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=220,
        hovermode="closest",
        legend=dict(bgcolor="rgba(0,0,0,0.3)", orientation="h",
                    x=0.5, y=-0.2, xanchor="center"),
        margin=dict(l=120, r=40, t=50, b=60),
    )
    fig.update_xaxes(gridcolor="#2a3441", showline=True, linecolor=DIM)
    fig.update_yaxes(
        categoryorder="array", categoryarray=list(reversed(REGIMES)),
        gridcolor="#2a3441", showgrid=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2: Per-regime average CAGR across top 25 runs
# ═══════════════════════════════════════════════════════════════════════════

def build_regime_avg_bars(runs_data: Dict[int, Dict],
                            classified_df: pd.DataFrame,
                            spy_curve: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Grouped bar chart showing, per regime:
      - Median annualized return across the top 25 runs
      - SPY annualized return in that regime (for reference)
      - Excess return (strategy - SPY)
    """
    if not runs_data or classified_df is None or classified_df.empty:
        return go.Figure()

    # Compute per-regime performance for each top run
    all_perf = []
    for run_id, data in runs_data.items():
        eq = data["equity"]
        perf = compute_regime_performance(eq, classified_df, spy_curve)
        if perf.empty:
            continue
        perf["run_id"] = run_id
        all_perf.append(perf)

    if not all_perf:
        return go.Figure()

    combined = pd.concat(all_perf, ignore_index=True)
    # Focus on main regimes only (drop Unknown)
    combined = combined[combined["Regime"].isin(REGIMES)]

    # Aggregate: median of annualized returns per regime across runs
    agg = combined.groupby("Regime").agg(
        median_ann=("Strategy_Annualized_Pct", "median"),
        q25_ann=("Strategy_Annualized_Pct", lambda x: x.quantile(0.25)),
        q75_ann=("Strategy_Annualized_Pct", lambda x: x.quantile(0.75)),
        median_spy_ann=("SPY_Annualized_Pct", "median"),
        n_days_median=("Days", "median"),
    ).reset_index()

    # Ensure canonical order
    order_map = {r: i for i, r in enumerate(REGIMES)}
    agg["order"] = agg["Regime"].map(order_map)
    agg = agg.sort_values("order").drop(columns="order").reset_index(drop=True)

    fig = go.Figure()

    # Strategy median return bars with error bars showing IQR
    fig.add_trace(go.Bar(
        x=agg["Regime"], y=agg["median_ann"],
        name="Strategy (median of top 25)",
        marker_color=[REGIME_COLORS_SOLID[r] for r in agg["Regime"]],
        error_y=dict(
            type="data",
            array=(agg["q75_ann"] - agg["median_ann"]).fillna(0),
            arrayminus=(agg["median_ann"] - agg["q25_ann"]).fillna(0),
            color=DIM, thickness=2, width=6,
        ),
        text=[f"{v:+.1f}%" for v in agg["median_ann"].fillna(0)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Strategy median: <b>%{y:+.1f}%</b> annualized"
                      "<br>IQR: [%{customdata[0]:+.1f}, %{customdata[1]:+.1f}]"
                      "<br>%{customdata[2]:.0f} median days<extra></extra>",
        customdata=np.column_stack([agg["q25_ann"].fillna(0),
                                      agg["q75_ann"].fillna(0),
                                      agg["n_days_median"].fillna(0)]),
    ))

    # SPY bars for comparison
    if "median_spy_ann" in agg.columns and not agg["median_spy_ann"].isna().all():
        fig.add_trace(go.Bar(
            x=agg["Regime"], y=agg["median_spy_ann"],
            name="SPY Buy & Hold",
            marker_color="rgba(140, 140, 140, 0.7)",
            marker_line_color="#aaa", marker_line_width=1,
            text=[f"{v:+.1f}%" for v in agg["median_spy_ann"].fillna(0)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>SPY: <b>%{y:+.1f}%</b> annualized<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=DIM)

    fig.update_layout(
        title=dict(text="Annualized Return by Market Regime  ·  "
                        "strategy median vs SPY",
                   font=dict(size=15, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        barmode="group",
        height=420,
        hovermode="closest",
        legend=dict(bgcolor="rgba(0,0,0,0.3)", x=0.01, y=0.99,
                    xanchor="left", yanchor="top"),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#2a3441")
    fig.update_yaxes(title_text="Annualized Return %",
                     gridcolor="#2a3441", ticksuffix="%",
                     zerolinecolor="#666")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: Heatmap — Top 25 runs × 4 regimes
# ═══════════════════════════════════════════════════════════════════════════

def build_regime_heatmap(runs_data: Dict[int, Dict],
                           classified_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap: rows = top runs (sorted by overall CAGR), columns = regimes.
    Cell = annualized return of that run within that regime.
    """
    if not runs_data or classified_df is None or classified_df.empty:
        return go.Figure()

    # Sort runs by overall CAGR desc
    sorted_runs = sorted(
        runs_data.items(),
        key=lambda x: x[1]["meta"].get("cagr_pct", 0) or 0,
        reverse=True,
    )

    rows = []
    row_labels = []
    for run_id, data in sorted_runs:
        perf = compute_regime_performance(data["equity"], classified_df)
        if perf.empty:
            continue
        perf_dict = {r: perf[perf["Regime"] == r]["Strategy_Annualized_Pct"].iloc[0]
                     if not perf[perf["Regime"] == r].empty else None
                     for r in REGIMES}
        rows.append([perf_dict.get(r) for r in REGIMES])
        cagr = data["meta"].get("cagr_pct", 0) or 0
        row_labels.append(f"#{run_id} (CAGR {cagr:+.1f}%)")

    if not rows:
        return go.Figure()

    z = np.array(rows, dtype=float)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=REGIMES, y=row_labels,
        colorscale=[
            [0.0, "#8B0000"], [0.4, "#2a3441"],
            [0.5, "#2a3441"], [0.6, "#2a3441"], [1.0, "#006400"]
        ],
        zmid=0, zmin=-40, zmax=60,
        text=z, texttemplate="%{text:.0f}%",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{y}<br><b>%{x}</b>: %{z:+.1f}% annualized<extra></extra>",
        colorbar=dict(title="Ann. %", tickfont=dict(color=TEXT)),
    ))

    fig.update_layout(
        title=dict(text="Per-Regime Annualized Return for Top 25 Runs",
                   font=dict(size=14, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=max(400, 30 + 22 * len(rows)),
        margin=dict(l=180, r=60, t=50, b=40),
    )
    fig.update_xaxes(side="top", tickangle=-15, gridcolor="#2a3441")
    fig.update_yaxes(autorange="reversed", gridcolor="#2a3441")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: Regime duration pie + stats table
# ═══════════════════════════════════════════════════════════════════════════

def build_regime_duration_chart(classified_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart: percentage of time spent in each regime.
    Provides quick context on how market conditions were distributed.
    """
    if classified_df is None or classified_df.empty:
        return go.Figure()

    counts = classified_df["Regime"].value_counts()
    # Canonical order
    order_map = {r: i for i, r in enumerate(REGIMES + ["Unknown"])}
    counts = counts.reindex([r for r in REGIMES + ["Unknown"] if r in counts.index])

    colors = [REGIME_COLORS_SOLID.get(r, "#6b7280") for r in counts.index]

    fig = go.Figure(data=go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value} days (%{percent})<extra></extra>",
    ))

    total_days = int(counts.sum())
    total_years = total_days / 252.0
    fig.update_layout(
        title=dict(text=f"Time Spent in Each Regime  ·  {total_years:.1f} years "
                        f"({total_days} trading days)",
                   font=dict(size=14, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=380,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        annotations=[
            dict(text=f"{total_years:.1f}y", x=0.5, y=0.5,
                 font=dict(size=28, color=TEXT), showarrow=False)
        ],
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main section builder
# ═══════════════════════════════════════════════════════════════════════════

def build_regime_section(db: MCDatabase, batch_id: str,
                           benchmarks_path: Optional[str] = None,
                           top_n: int = 25) -> Dict[str, str]:
    """
    Build the Regime Analysis section for the dashboard.
    Returns dict of HTML divs.
    """
    if not benchmarks_path or not Path(benchmarks_path).exists():
        no_data = _empty_div("Regime analysis requires benchmarks.parquet with SPY+VIX data")
        return {
            "regime_legend": "",
            "regime_timeline": no_data,
            "regime_avg_bars": no_data,
            "regime_heatmap": no_data,
            "regime_duration": no_data,
            "regime_summary": "",
        }

    try:
        bench_df = pd.read_parquet(benchmarks_path)
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        err = _empty_div(f"Failed to load benchmarks: {e}")
        return {k: err for k in ["regime_legend", "regime_timeline",
                                  "regime_avg_bars", "regime_heatmap",
                                  "regime_duration", "regime_summary"]}

    runs_data = load_top_runs_with_curves(db, batch_id, top_n=top_n)
    if not runs_data:
        no_runs = _empty_div("No runs with detail data")
        return {k: no_runs for k in ["regime_legend", "regime_timeline",
                                       "regime_avg_bars", "regime_heatmap",
                                       "regime_duration", "regime_summary"]}

    # Classify
    classified = classify_regimes(bench_df)

    # Restrict to strategy date range
    first_eq = next(iter(runs_data.values()))["equity"]
    first_dates = pd.to_datetime(first_eq["Date"])
    date_min, date_max = first_dates.min(), first_dates.max()
    classified_range = classified[
        (classified["Date"] >= date_min) & (classified["Date"] <= date_max)
    ].reset_index(drop=True)

    # SPY curve for comparison
    initial_capital = float(first_eq["Capital"].iloc[0])
    spy_curve = load_spy_curve(benchmarks_path, first_eq["Date"], initial_capital)

    charts = {
        "regime_legend": regime_legend_annotations(),
        "regime_timeline": _fig_to_div(build_regime_timeline(classified_range)),
        "regime_avg_bars": _fig_to_div(build_regime_avg_bars(runs_data, classified_range, spy_curve)),
        "regime_heatmap": _fig_to_div(build_regime_heatmap(runs_data, classified_range)),
        "regime_duration": _fig_to_div(build_regime_duration_chart(classified_range)),
    }

    # Summary box
    counts = classified_range["Regime"].value_counts()
    crisis_days = int(counts.get("Crisis", 0))
    correction_days = int(counts.get("Correction", 0))
    bear_pct = (crisis_days + correction_days) / len(classified_range) * 100 \
               if len(classified_range) > 0 else 0

    charts["regime_summary"] = f"""
    <div style="color: {DIM}; padding: 10px 14px; background: {CARD};
                border-radius: 6px; margin-bottom: 14px; font-size: 13px;">
      <strong style="color: {TEXT};">Market Regime Analysis</strong> — the market
      spent <strong style="color: {NEG};">{bear_pct:.1f}%</strong> of the
      20-year period in Correction or Crisis regimes. This tab answers:
      <em>is the strategy's alpha consistent across regimes, or does it
      come from one specific regime (bull calm)?</em>
    </div>
    """

    return charts
