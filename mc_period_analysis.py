"""
mc_period_analysis.py — Period-based analysis of top MC runs.

Five analyses to understand WHEN and HOW strategies generate alpha:

  A. Rolling 1-Year CAGR        — is performance stable or driven by spikes?
  B. Excess Return vs SPY       — when does the strategy beat the benchmark?
  C. Yearly Returns Heatmap     — year-by-year breakdown
  D. Drawdown Periods           — depth and duration of underwater spells
  E. Strategy Cross-Correlation — are top N runs really N strategies, or just 1?

All charts return Plotly Figure objects ready for HTML embedding.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mc_database import MCDatabase

logger = logging.getLogger("mc_period_analysis")

# Shared colors (match mc_dashboard)
BG        = "#0f1419"
CARD      = "#1a2332"
TEXT      = "#e6edf3"
DIM       = "#8b949e"
POS       = "#2ea043"
NEG       = "#f85149"
SPY_COLOR = "#95A5A6"


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helper
# ══════════════════════════════════════════════════════════════════════════════

def load_top_runs_with_curves(db: MCDatabase, batch_id: str,
                                top_n: int = 25) -> Dict[int, Dict]:
    """
    Load top N runs from DB, returning only those with detail data.
    Returns: {run_id: {'meta': {...}, 'equity': pd.DataFrame}}
    """
    runs = db.query_runs(batch_id=batch_id, order_by="cagr_pct", limit=top_n)
    if runs.empty:
        return {}

    result = {}
    skipped_no_detail = 0
    for _, row in runs.iterrows():
        run_id = int(row["run_id"])
        if not row.get("has_detail_data"):
            skipped_no_detail += 1
            continue
        details = db.get_run_details(run_id)
        eq = details.get("equity_curve")
        if eq is None or eq.empty:
            continue
        eq = eq.sort_values("Date").reset_index(drop=True)
        eq["Date"] = pd.to_datetime(eq["Date"])
        result[run_id] = {
            "meta": {
                "run_id": run_id,
                "cagr_pct": row.get("cagr_pct"),
                "max_dd_pct": row.get("max_drawdown_pct"),
                "alpha_vs_spy": row.get("alpha_vs_spy"),
                "params": row.get("params", {}),
            },
            "equity": eq,
        }

    if skipped_no_detail > 0:
        logger.info(f"Skipped {skipped_no_detail} runs without detail data "
                    f"(only top/bottom from MC have it)")
    logger.info(f"Loaded {len(result)} runs with full equity curves")
    return result


def load_spy_curve(benchmarks_path: str, dates: pd.Series,
                    initial_capital: float) -> Optional[pd.DataFrame]:
    """Build SPY Buy&Hold curve aligned to given dates."""
    try:
        b = pd.read_parquet(benchmarks_path)
    except Exception as e:
        logger.warning(f"benchmarks load failed: {e}")
        return None

    b["Date"] = pd.to_datetime(b["Date"]).dt.normalize()
    if "SPY_Adj_Close" not in b.columns:
        return None

    dates_set = set(pd.to_datetime(dates).dt.normalize())
    sub = b[b["Date"].isin(dates_set)][["Date", "SPY_Adj_Close"]].copy()
    sub["SPY_Adj_Close"] = sub["SPY_Adj_Close"].ffill()
    if sub.empty or sub["SPY_Adj_Close"].iloc[0] <= 0:
        return None
    shares = initial_capital / sub["SPY_Adj_Close"].iloc[0]
    sub["SPY_Capital"] = sub["SPY_Adj_Close"] * shares
    return sub[["Date", "SPY_Capital"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# A. Rolling 1-Year CAGR
# ══════════════════════════════════════════════════════════════════════════════

def build_rolling_cagr_chart(runs_data: Dict[int, Dict],
                               spy_curve: Optional[pd.DataFrame] = None,
                               window_days: int = 252,
                               regime_shapes: Optional[List[Dict]] = None) -> go.Figure:
    """
    For each run, plot the rolling 1-year (252 trading days) CAGR.
    A flat line near 20% means stable performance.
    A line spiking from -30 to +60 means lumpy, unreliable returns.
    """
    fig = go.Figure()

    if not runs_data:
        return _empty_fig("No runs with equity curves")

    for run_id, data in runs_data.items():
        eq = data["equity"]
        cap = eq["Capital"].astype(float).values
        if len(cap) < window_days + 1:
            continue
        # Rolling % change over `window_days` trading days
        roll_ret = pd.Series(cap).pct_change(window_days) * 100
        meta = data["meta"]
        label = (f"#{run_id} — CAGR {meta.get('cagr_pct', 0):.1f}%, "
                 f"top_n={meta.get('params', {}).get('top_n')}")
        fig.add_trace(go.Scatter(
            x=eq["Date"], y=roll_ret, name=label,
            mode="lines", line=dict(width=1.2),
            opacity=0.7,
            hovertemplate=f"<b>Run #{run_id}</b><br>"
                          "%{x|%Y-%m-%d}<br>1Y return: %{y:+.1f}%<extra></extra>",
        ))

    if spy_curve is not None and len(spy_curve) >= window_days + 1:
        spy_roll = spy_curve["SPY_Capital"].pct_change(window_days) * 100
        fig.add_trace(go.Scatter(
            x=spy_curve["Date"], y=spy_roll, name="SPY (1Y)",
            mode="lines", line=dict(color=SPY_COLOR, width=2.5, dash="dot"),
            hovertemplate="<b>SPY</b><br>%{x|%Y-%m-%d}<br>1Y: %{y:+.1f}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=DIM, line_width=1)
    layout_args = dict(
        title=dict(text=f"Rolling {window_days // 21}-Month Return — "
                        "are returns stable or lumpy?",
                   font=dict(size=15, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=480,
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.99,
                    xanchor="left", yanchor="top", font=dict(size=10)),
        margin=dict(l=60, r=40, t=50, b=40),
    )
    if regime_shapes:
        layout_args["shapes"] = regime_shapes
    fig.update_layout(**layout_args)
    fig.update_xaxes(gridcolor="#2a3441")
    fig.update_yaxes(title_text="Trailing 1Y Return %", gridcolor="#2a3441",
                     ticksuffix="%", zerolinecolor="#666")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# B. Excess Return vs SPY (rolling)
# ══════════════════════════════════════════════════════════════════════════════

def build_excess_return_chart(runs_data: Dict[int, Dict],
                                spy_curve: pd.DataFrame,
                                window_days: int = 63,
                                regime_shapes: Optional[List[Dict]] = None) -> go.Figure:
    """
    Rolling 3-month excess return: strategy_3M_return - SPY_3M_return.
    Positive area = strategy beating SPY in this window.
    Negative area = SPY beating strategy.

    Helps answer: when does the strategy add value? Bear markets? Bull?
    """
    fig = go.Figure()

    if not runs_data or spy_curve is None or spy_curve.empty:
        return _empty_fig("No data for excess return chart")

    # Synthesize a "Top 10 portfolio" by averaging returns of best 10
    sorted_runs = sorted(runs_data.items(),
                         key=lambda x: x[1]["meta"].get("cagr_pct", 0) or 0,
                         reverse=True)
    top10 = dict(sorted_runs[:10])

    # Build composite returns: average of top10 daily returns at each date
    composite_rows = []
    for run_id, data in top10.items():
        eq = data["equity"][["Date", "Capital"]].copy()
        eq["Date"] = pd.to_datetime(eq["Date"]).dt.normalize()
        eq["ret"] = eq["Capital"].pct_change()
        eq["run_id"] = run_id
        composite_rows.append(eq[["Date", "ret", "run_id"]])

    if not composite_rows:
        return _empty_fig("No equity curves available")

    long_df = pd.concat(composite_rows, ignore_index=True)
    avg_returns = long_df.groupby("Date")["ret"].mean().reset_index()
    avg_returns["roll_ret"] = (1 + avg_returns["ret"]).rolling(window_days).apply(
        lambda x: x.prod() - 1, raw=True
    ) * 100

    # SPY rolling
    spy = spy_curve.copy()
    spy["Date"] = pd.to_datetime(spy["Date"]).dt.normalize()
    spy["ret"] = spy["SPY_Capital"].pct_change()
    spy["roll_ret"] = (1 + spy["ret"]).rolling(window_days).apply(
        lambda x: x.prod() - 1, raw=True
    ) * 100

    merged = avg_returns.merge(spy[["Date", "roll_ret"]], on="Date",
                                how="inner", suffixes=("_strat", "_spy"))
    merged["excess"] = merged["roll_ret_strat"] - merged["roll_ret_spy"]
    merged = merged.dropna(subset=["excess"])

    if merged.empty:
        return _empty_fig("No overlapping dates with SPY")

    pos_mask = merged["excess"] >= 0
    neg_mask = ~pos_mask

    fig.add_trace(go.Scatter(
        x=merged.loc[pos_mask, "Date"], y=merged.loc[pos_mask, "excess"],
        name="Strategy beats SPY", mode="lines",
        line=dict(color=POS, width=0), fill="tozeroy",
        fillcolor="rgba(46, 160, 67, 0.5)",
        hovertemplate="%{x|%Y-%m-%d}<br>Excess: <b>+%{y:.1f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=merged.loc[neg_mask, "Date"], y=merged.loc[neg_mask, "excess"],
        name="SPY beats Strategy", mode="lines",
        line=dict(color=NEG, width=0), fill="tozeroy",
        fillcolor="rgba(248, 81, 73, 0.5)",
        hovertemplate="%{x|%Y-%m-%d}<br>Lag: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=DIM)

    # summary stats
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    pct_outperform = n_pos / (n_pos + n_neg) * 100 if (n_pos + n_neg) > 0 else 0
    avg_excess = float(merged["excess"].mean())

    layout_args = dict(
        title=dict(text=f"Top 10 Composite — 3-Month Excess Return vs SPY  ·  "
                        f"outperforms {pct_outperform:.0f}% of time, avg "
                        f"{avg_excess:+.1f}%/quarter",
                   font=dict(size=14, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=400,
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.99,
                    xanchor="left", yanchor="top"),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    if regime_shapes:
        layout_args["shapes"] = regime_shapes
    fig.update_layout(**layout_args)
    fig.update_xaxes(gridcolor="#2a3441")
    fig.update_yaxes(title_text="Excess Return %", gridcolor="#2a3441",
                     ticksuffix="%", zerolinecolor="#666")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# C. Yearly Returns Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def build_yearly_heatmap(runs_data: Dict[int, Dict],
                           spy_curve: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Year × Run heatmap of annual returns.
    Last column shows SPY for reference.
    Quickly reveals which years drove the CAGR.
    """
    if not runs_data:
        return _empty_fig("No runs")

    rows = []
    for run_id, data in runs_data.items():
        eq = data["equity"][["Date", "Capital"]].copy()
        eq["Date"] = pd.to_datetime(eq["Date"])
        eq["Year"] = eq["Date"].dt.year
        # last value per year, and previous-year-end as base
        yearly_last = eq.groupby("Year")["Capital"].last()
        yearly_first = eq.groupby("Year")["Capital"].first()
        for year in yearly_last.index:
            ret = (yearly_last[year] / yearly_first[year] - 1) * 100
            rows.append({"Year": year, "RunOrSPY": f"#{run_id}", "Return": ret})

    if spy_curve is not None and not spy_curve.empty:
        spy = spy_curve.copy()
        spy["Year"] = spy["Date"].dt.year
        sp_last = spy.groupby("Year")["SPY_Capital"].last()
        sp_first = spy.groupby("Year")["SPY_Capital"].first()
        for year in sp_last.index:
            rows.append({
                "Year": year, "RunOrSPY": "SPY",
                "Return": (sp_last[year] / sp_first[year] - 1) * 100,
            })

    long_df = pd.DataFrame(rows)
    pivot = long_df.pivot_table(index="Year", columns="RunOrSPY", values="Return")

    # Order columns: runs sorted by CAGR desc, then SPY at end
    sorted_run_ids = sorted(
        runs_data.keys(),
        key=lambda r: runs_data[r]["meta"].get("cagr_pct", 0) or 0,
        reverse=True
    )
    col_order = [f"#{r}" for r in sorted_run_ids]
    if "SPY" in pivot.columns:
        col_order.append("SPY")
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, "#8B0000"], [0.5, "#2a3441"], [1, "#006400"]],
        zmid=0,
        zmin=-50, zmax=80,
        text=pivot.values,
        texttemplate="%{text:.0f}",
        textfont=dict(size=9, color="white"),
        hovertemplate="%{x}<br>%{y}: <b>%{z:.1f}%</b><extra></extra>",
        colorbar=dict(title="%", tickfont=dict(color=TEXT)),
    ))

    fig.update_layout(
        title=dict(text="Yearly Returns — top runs sorted left-to-right by CAGR, SPY at far right",
                   font=dict(size=14, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=max(400, 35 + 22 * pivot.shape[0]),
        margin=dict(l=60, r=60, t=50, b=50),
    )
    fig.update_xaxes(side="top", gridcolor="#2a3441", tickangle=-45)
    fig.update_yaxes(autorange="reversed", dtick=1, gridcolor="#2a3441")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# D. Drawdown Periods
# ══════════════════════════════════════════════════════════════════════════════

def build_drawdown_underwater(runs_data: Dict[int, Dict],
                                 spy_curve: Optional[pd.DataFrame] = None,
                                 max_lines: int = 10,
                                 regime_shapes: Optional[List[Dict]] = None) -> go.Figure:
    """
    Underwater drawdown plot for top N runs + SPY.
    Reveals when (and how long) strategies were below their high water mark.
    """
    if not runs_data:
        return _empty_fig("No runs")

    fig = go.Figure()
    sorted_runs = sorted(
        runs_data.items(),
        key=lambda x: x[1]["meta"].get("cagr_pct", 0) or 0,
        reverse=True
    )[:max_lines]

    for run_id, data in sorted_runs:
        eq = data["equity"]
        cap = eq["Capital"].astype(float)
        peak = cap.cummax()
        dd = (cap - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=eq["Date"], y=dd, name=f"#{run_id}",
            mode="lines", line=dict(width=1.2), opacity=0.6,
            hovertemplate=f"<b>Run #{run_id}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
        ))

    if spy_curve is not None and not spy_curve.empty:
        cap = spy_curve["SPY_Capital"].astype(float)
        peak = cap.cummax()
        dd = (cap - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=spy_curve["Date"], y=dd, name="SPY",
            mode="lines", line=dict(color=SPY_COLOR, width=2.5, dash="dot"),
            hovertemplate="<b>SPY</b><br>%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
        ))

    layout_args = dict(
        title=dict(text=f"Underwater Drawdown — top {len(sorted_runs)} runs vs SPY",
                   font=dict(size=14, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=380,
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0.4)", x=0.01, y=0.01,
                    xanchor="left", yanchor="bottom", font=dict(size=10)),
        margin=dict(l=60, r=40, t=50, b=40),
    )
    if regime_shapes:
        layout_args["shapes"] = regime_shapes
    fig.update_layout(**layout_args)
    fig.update_xaxes(gridcolor="#2a3441")
    fig.update_yaxes(title_text="Drawdown %", gridcolor="#2a3441",
                     ticksuffix="%", range=[-100, 5])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E. Strategy Cross-Correlation
# ══════════════════════════════════════════════════════════════════════════════

def build_correlation_matrix(runs_data: Dict[int, Dict]) -> go.Figure:
    """
    Daily-return correlation matrix between top N runs.
    All cells dark green (~1.0) = strategies are essentially the same.
    Mixed colors = strategies make different bets, ensemble could be useful.
    """
    if not runs_data or len(runs_data) < 2:
        return _empty_fig("Need ≥2 runs for correlation matrix")

    sorted_runs = sorted(
        runs_data.items(),
        key=lambda x: x[1]["meta"].get("cagr_pct", 0) or 0,
        reverse=True
    )

    # Build wide matrix of daily returns
    series = {}
    for run_id, data in sorted_runs:
        eq = data["equity"][["Date", "Capital"]].copy()
        eq["Date"] = pd.to_datetime(eq["Date"]).dt.normalize()
        ret = eq.set_index("Date")["Capital"].pct_change()
        series[f"#{run_id}"] = ret

    wide = pd.DataFrame(series).dropna()
    if wide.empty:
        return _empty_fig("No overlapping returns")

    corr = wide.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdYlGn", zmin=0.5, zmax=1.0,
        text=corr.values, texttemplate="%{text:.2f}",
        textfont=dict(size=9, color="black"),
        hovertemplate="%{y} ↔ %{x}: <b>%{z:.3f}</b><extra></extra>",
        colorbar=dict(title="ρ", tickfont=dict(color=TEXT)),
    ))

    avg_corr = float(corr.values[np.triu_indices_from(corr, k=1)].mean())

    fig.update_layout(
        title=dict(text=f"Top {len(corr)} Cross-Correlation — "
                        f"avg ρ = {avg_corr:.2f}  ·  "
                        f"({'highly redundant' if avg_corr > 0.9 else 'diverse strategies' if avg_corr < 0.7 else 'moderately related'})",
                   font=dict(size=14, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=max(400, 30 + 22 * len(corr)),
        margin=dict(l=80, r=60, t=50, b=80),
    )
    fig.update_xaxes(side="bottom", tickangle=-45)
    fig.update_yaxes(autorange="reversed")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                        xref="paper", yref="paper", x=0.5, y=0.5,
                        font=dict(size=16, color=DIM))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        height=300, margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Aggregator: build all 5 charts and return as HTML divs
# ══════════════════════════════════════════════════════════════════════════════

def build_period_analysis_section(db: MCDatabase, batch_id: str,
                                    benchmarks_path: Optional[str] = None,
                                    top_n: int = 25) -> Dict[str, str]:
    """
    Build all five period-analysis charts, with regime shading from VIX+SPY.
    Returns dict of {chart_id: html_div_string} for embedding in dashboard.
    """
    from plotly.offline import plot as plotly_plot

    def fig_to_div(fig: go.Figure) -> str:
        return plotly_plot(fig, include_plotlyjs=False, output_type="div",
                            config={"responsive": True, "displaylogo": False})

    runs_data = load_top_runs_with_curves(db, batch_id, top_n=top_n)
    if not runs_data:
        empty = '<div class="empty-state">No runs with detail data found. ' \
                'Re-run MC to generate top/bottom detail.</div>'
        return {key: empty for key in ["rolling_cagr", "excess_return",
                                         "yearly_heatmap", "drawdown",
                                         "correlation", "summary", "regime_legend"]}

    # Get common date range from first run for SPY curve
    first_eq = next(iter(runs_data.values()))["equity"]
    initial_capital = float(first_eq["Capital"].iloc[0])
    spy_curve = None
    regime_shapes = None
    regime_legend_html = ""
    if benchmarks_path:
        spy_curve = load_spy_curve(benchmarks_path, first_eq["Date"], initial_capital)

        # Build regime shapes for background shading
        try:
            from mc_regime import (classify_regimes, regime_shapes_for_plotly,
                                    regime_legend_annotations)
            bench_df = pd.read_parquet(benchmarks_path)
            classified = classify_regimes(bench_df)
            # Restrict to the strategy's date range
            first_dates = pd.to_datetime(first_eq["Date"])
            date_min = first_dates.min()
            date_max = first_dates.max()
            classified = classified[
                (classified["Date"] >= date_min) & (classified["Date"] <= date_max)
            ]
            regime_shapes = regime_shapes_for_plotly(classified, min_days=15,
                                                       opacity_scale=0.8)
            regime_legend_html = regime_legend_annotations()
            logger.info(f"Added regime shading: {len(regime_shapes)} regions")
        except Exception as e:
            logger.warning(f"Regime shading failed (non-fatal): {e}")
            regime_shapes = None

    charts = {}
    charts["rolling_cagr"] = fig_to_div(
        build_rolling_cagr_chart(runs_data, spy_curve, regime_shapes=regime_shapes)
    )
    if spy_curve is not None:
        charts["excess_return"] = fig_to_div(
            build_excess_return_chart(runs_data, spy_curve, regime_shapes=regime_shapes)
        )
    else:
        charts["excess_return"] = '<div class="empty-state">Excess return chart needs benchmarks.parquet</div>'
    charts["yearly_heatmap"] = fig_to_div(build_yearly_heatmap(runs_data, spy_curve))
    charts["drawdown"] = fig_to_div(
        build_drawdown_underwater(runs_data, spy_curve, max_lines=10,
                                     regime_shapes=regime_shapes)
    )
    charts["correlation"] = fig_to_div(build_correlation_matrix(runs_data))

    # Regime legend (HTML snippet for display above charts)
    charts["regime_legend"] = regime_legend_html

    # Header summary
    n_runs = len(runs_data)
    avg_cagr = np.mean([d["meta"].get("cagr_pct", 0) or 0 for d in runs_data.values()])
    summary = f"""
    <div style="color: {DIM}; padding: 8px 14px; background: {CARD};
                border-radius: 6px; margin-bottom: 14px; font-size: 13px;">
      <strong style="color: {TEXT};">Period Analysis</strong> — analyzing
      {n_runs} top runs with full equity data (avg CAGR
      <span style="color: {POS};">{avg_cagr:+.2f}%</span>).
      Charts answer: <em>is performance stable, when does it beat SPY,
      which years drove the returns, drawdown patterns,
      are the top runs really diverse strategies?</em>
      <br><em>Background shading</em>: market regime based on SPY trend + VIX level.
    </div>
    """
    charts["summary"] = summary

    return charts
