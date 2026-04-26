"""
mc_dashboard.py — Interactive HTML dashboard for an MC batch.

Different from the single-strategy dashboard (reporting.py):
  - Shows the distribution of ALL runs (scatter: CAGR vs MaxDD, colored by Sharpe)
  - Parameter sensitivity: which params matter most?
  - Top 25 table with click-to-expand equity curve
  - Period comparison — pick a date range, see how runs performed in that window
  - Each run can be individually compared vs SPY/QQQ

Output: single standalone HTML file with Plotly embedded.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot

from mc_database import MCDatabase

logger = logging.getLogger("mc_dashboard")

# Colors
BG        = "#0f1419"
CARD      = "#1a2332"
TEXT      = "#e6edf3"
DIM       = "#8b949e"
POS       = "#2ea043"
NEG       = "#f85149"
NEUTRAL   = "#3498DB"
SPY_COLOR = "#95A5A6"
QQQ_COLOR = "#7F8C8D"


# ══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════════════

def build_scatter_runs(runs_df: pd.DataFrame) -> str:
    """Scatter plot: x=MaxDD, y=CAGR, color=Sharpe, size=trades. Hover=params."""
    if runs_df.empty:
        return '<div class="empty-state">No runs to display</div>'

    df = runs_df.dropna(subset=["cagr_pct", "max_drawdown_pct"]).copy()
    hover_texts = []
    for _, r in df.iterrows():
        params_str = "<br>".join(f"  {k}: {v}" for k, v in r.get("params", {}).items())
        hover_texts.append(
            f"<b>Run #{r['run_id']}</b><br>"
            f"CAGR: {r['cagr_pct']:+.2f}%<br>"
            f"MaxDD: {r['max_drawdown_pct']:.1f}%<br>"
            f"Sharpe: {r.get('sharpe', 0):.2f}<br>"
            f"Trades: {r.get('total_trades', 0)}<br>"
            f"<br><b>Params:</b><br>{params_str}"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["max_drawdown_pct"], y=df["cagr_pct"],
        mode="markers",
        marker=dict(
            size=np.clip(df["total_trades"].fillna(10) / 50, 6, 18),
            color=df["sharpe"].fillna(0),
            colorscale="RdYlGn", cmin=-1, cmax=2.5,
            showscale=True, colorbar=dict(title="Sharpe"),
            line=dict(color="rgba(255,255,255,0.3)", width=0.5),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        customdata=df["run_id"],
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=DIM, line_width=1)
    fig.update_layout(
        title=dict(text="All Runs — CAGR vs MaxDD (color = Sharpe)",
                   font=dict(size=16, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=550,
        margin=dict(l=60, r=40, t=50, b=50),
    )
    fig.update_xaxes(title_text="Max Drawdown %", gridcolor="#2a3441",
                     ticksuffix="%", zerolinecolor="#2a3441")
    fig.update_yaxes(title_text="CAGR %", gridcolor="#2a3441",
                     ticksuffix="%", zerolinecolor="#2a3441")
    return _fig_to_div(fig, elem_id="scatter-runs")


def build_param_sensitivity(runs_df: pd.DataFrame) -> str:
    """For each param, show CAGR distribution across its values."""
    if runs_df.empty:
        return '<div class="empty-state">No runs</div>'

    df = runs_df.dropna(subset=["cagr_pct"]).copy()
    if "params" not in df.columns:
        return '<div class="empty-state">No params column</div>'

    # Expand params to columns
    param_cols: Dict[str, List] = {}
    for _, r in df.iterrows():
        for k, v in r["params"].items():
            param_cols.setdefault(k, [None] * len(df))

    # Build long-form data: one row per (run, param) pair
    rows = []
    for idx, (_, r) in enumerate(df.iterrows()):
        for k, v in r["params"].items():
            rows.append({
                "run_id": r["run_id"],
                "param": k,
                "value": v,
                "cagr_pct": r["cagr_pct"],
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return '<div class="empty-state">No param data</div>'

    # For each param, compute correlation with CAGR
    summary_rows = []
    for param in long_df["param"].unique():
        sub = long_df[long_df["param"] == param]
        # If numeric — correlation
        try:
            vals = pd.to_numeric(sub["value"])
            if vals.std() > 0:
                corr = vals.corr(sub["cagr_pct"])
            else:
                corr = 0.0
        except (ValueError, TypeError):
            # Categorical — use mean CAGR range across categories
            grouped = sub.groupby("value")["cagr_pct"].mean()
            corr = float(grouped.max() - grouped.min()) / 10  # normalized proxy
        summary_rows.append({
            "param": param,
            "correlation": corr,
            "cagr_range": f"{sub['cagr_pct'].min():+.1f}% to {sub['cagr_pct'].max():+.1f}%",
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("correlation", key=abs, ascending=False)

    # Bar chart of |correlation|
    fig = go.Figure()
    colors = [POS if v > 0 else NEG for v in summary_df["correlation"]]
    fig.add_trace(go.Bar(
        x=summary_df["correlation"],
        y=summary_df["param"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>corr with CAGR: %{x:+.2f}<extra></extra>",
        text=[f"{c:+.2f}" for c in summary_df["correlation"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Parameter Sensitivity — Correlation with CAGR",
                   font=dict(size=15, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=max(300, 40 + 35 * len(summary_df)),
        margin=dict(l=140, r=60, t=50, b=40),
    )
    fig.update_xaxes(title_text="Correlation", gridcolor="#2a3441",
                     zerolinecolor="#eeeeee", range=[-1, 1])
    fig.update_yaxes(gridcolor="#2a3441", autorange="reversed")
    return _fig_to_div(fig, elem_id="param-sens")


def build_equity_overlay(db: MCDatabase, run_ids: List[int],
                          benchmarks_path: Optional[str]) -> str:
    """
    Overlay plot: equity curves of top runs + SPY + QQQ on one chart.
    Only runs with detail data (equity saved) can be shown.
    """
    traces = []
    first_eq = None
    for rid in run_ids:
        try:
            details = db.get_run_details(rid)
        except Exception:
            continue
        eq = details.get("equity_curve")
        if eq is None or eq.empty:
            continue
        if first_eq is None:
            first_eq = eq
        label = f"Run #{rid} ({details.get('cagr_pct', 0):.1f}% CAGR)"
        traces.append(go.Scatter(
            x=eq["Date"], y=eq["Capital"], name=label,
            mode="lines", line=dict(width=1.6),
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
        ))

    if not traces or first_eq is None:
        return '<div class="empty-state">No equity curves available (Top 25 needed)</div>'

    # Add benchmarks
    if benchmarks_path and Path(benchmarks_path).exists():
        try:
            b = pd.read_parquet(benchmarks_path)
            b["Date"] = pd.to_datetime(b["Date"])
            dates = pd.to_datetime(first_eq["Date"])
            initial = float(first_eq["Capital"].iloc[0])
            for col_prefix, color, style in [("SPY", SPY_COLOR, "dot"),
                                              ("QQQ", QQQ_COLOR, "dash")]:
                adj_col = f"{col_prefix}_Adj_Close"
                if adj_col not in b.columns:
                    continue
                bsub = b[b["Date"].isin(dates)][["Date", adj_col]].copy()
                bsub[adj_col] = bsub[adj_col].ffill()
                if bsub.empty or bsub[adj_col].iloc[0] <= 0:
                    continue
                shares = initial / bsub[adj_col].iloc[0]
                bsub["cap"] = bsub[adj_col] * shares
                traces.append(go.Scatter(
                    x=bsub["Date"], y=bsub["cap"],
                    name=f"{col_prefix} B&H", mode="lines",
                    line=dict(color=color, width=1.8, dash=style),
                    hovertemplate=f"<b>{col_prefix}</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
                ))
        except Exception as e:
            logger.debug(f"benchmark overlay failed: {e}")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"Equity Curves — Top {len(run_ids)} Runs vs Benchmarks (log scale)",
                   font=dict(size=16, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=550,
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0.3)", x=0.01, y=0.99,
                    xanchor="left", yanchor="top"),
        margin=dict(l=60, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#2a3441")
    fig.update_yaxes(title_text="Capital ($)", type="log",
                     tickformat="$,.0f", gridcolor="#2a3441")
    return _fig_to_div(fig, elem_id="equity-overlay")


def build_cagr_histogram(runs_df: pd.DataFrame) -> str:
    """Histogram of CAGR distribution."""
    if runs_df.empty:
        return '<div class="empty-state">No runs</div>'
    df = runs_df.dropna(subset=["cagr_pct"])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["cagr_pct"],
        nbinsx=min(50, max(10, int(len(df) / 10))),
        marker_color=NEUTRAL,
        hovertemplate="CAGR range: %{x}<br>Runs: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=DIM)
    # Add median
    median_cagr = df["cagr_pct"].median()
    fig.add_vline(x=median_cagr, line_dash="dash", line_color=POS,
                  annotation_text=f"Median {median_cagr:+.1f}%")

    fig.update_layout(
        title=dict(text=f"CAGR Distribution — {len(df)} Runs",
                   font=dict(size=15, color=TEXT)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT),
        height=350,
        margin=dict(l=60, r=40, t=50, b=40),
        bargap=0.05,
    )
    fig.update_xaxes(title_text="CAGR %", gridcolor="#2a3441", ticksuffix="%")
    fig.update_yaxes(title_text="Runs count", gridcolor="#2a3441")
    return _fig_to_div(fig, elem_id="cagr-hist")


def _fig_to_div(fig, elem_id: str = "") -> str:
    html = plotly_plot(fig, include_plotlyjs=False, output_type="div",
                       config={"responsive": True, "displaylogo": False})
    # Inject custom id on the outer div so click handlers can find it
    if elem_id and 'class="plotly-graph-div"' in html:
        # The generated HTML has e.g. <div id="abc..." class="plotly-graph-div"...>
        # We inject a wrapper div with our id
        html = f'<div id="{elem_id}">{html}</div>'
    return html


# ══════════════════════════════════════════════════════════════════════════════
# HTML table builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_top_table(runs_df: pd.DataFrame, n: int = 25) -> str:
    """Table of top N runs with expandable params."""
    if runs_df.empty:
        return '<div class="empty-state">No runs</div>'

    top = runs_df.head(n)
    rows_html = []
    for _, r in top.iterrows():
        params_items = r.get("params", {}) or {}
        params_str = " · ".join(f"<code>{k}={v}</code>" for k, v in params_items.items())

        cagr_v = r.get("cagr_pct", 0) or 0
        dd_v   = r.get("max_drawdown_pct", 0) or 0
        sharpe_v = r.get("sharpe", 0) or 0
        alpha_v = r.get("alpha_vs_spy", 0) or 0
        trades_v = r.get("total_trades", 0) or 0

        cagr_color = POS if cagr_v > 10 else NEG if cagr_v < 0 else TEXT
        alpha_color = POS if alpha_v > 0 else NEG
        has_detail = "✓" if r.get("has_detail_data") else "—"

        rows_html.append(f"""
        <tr>
          <td class="run-id">#{r['run_id']}</td>
          <td style="color: {cagr_color}; font-weight: 600;">{cagr_v:+.2f}%</td>
          <td>{dd_v:.1f}%</td>
          <td>{sharpe_v:.2f}</td>
          <td style="color: {alpha_color};">{alpha_v:+.2f}%</td>
          <td>{int(trades_v):,}</td>
          <td>{r.get('win_rate_pct', 0):.1f}%</td>
          <td style="text-align:center;">{has_detail}</td>
          <td class="params-cell">{params_str}</td>
        </tr>
        """)

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Run ID</th>
          <th>CAGR</th>
          <th>MaxDD</th>
          <th>Sharpe</th>
          <th>Alpha vs SPY</th>
          <th>Trades</th>
          <th>Win%</th>
          <th>Detail</th>
          <th>Parameters</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """


def _build_batch_summary(batch: Dict, runs_df: pd.DataFrame) -> str:
    """KPI cards with batch summary."""
    n_total = len(runs_df)
    if n_total == 0:
        return '<div class="empty-state">No completed runs</div>'

    cagr_values = runs_df["cagr_pct"].dropna()
    dd_values = runs_df["max_drawdown_pct"].dropna()
    winners = (cagr_values > 0).sum()
    beats_spy = ((runs_df.get("excess_cagr_spy", 0).dropna()) > 0).sum()

    cards = [
        ("Total Runs", f"{n_total:,}", batch.get("strategy_name", ""), None),
        ("Best CAGR", f"{cagr_values.max():+.2f}%", f"worst: {cagr_values.min():+.1f}%", True),
        ("Median CAGR", f"{cagr_values.median():+.2f}%", "middle of distribution", cagr_values.median() > 10),
        ("Profitable Runs", f"{winners}/{n_total}", f"{winners/n_total*100:.1f}%", winners > n_total/2),
        ("Beat SPY", f"{beats_spy}/{n_total}", f"{beats_spy/n_total*100:.1f}% of runs", beats_spy > n_total/3),
        ("Best MaxDD", f"{dd_values.max():.1f}%", f"worst: {dd_values.min():.1f}%", dd_values.max() > -25),
    ]

    cards_html = []
    for label, value, sub, positive in cards:
        color = POS if positive is True else NEG if positive is False else TEXT
        cards_html.append(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color: {color};">{value}</div>
          <div class="kpi-subtext">{sub}</div>
        </div>
        """)
    return f'<div class="kpi-row">{"".join(cards_html)}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# Main HTML template
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MC Dashboard — Batch {batch_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    background: {bg}; color: {text};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 20px;
  }}
  header {{ margin-bottom: 18px; border-bottom: 1px solid #2a3441; padding-bottom: 12px; }}
  h1 {{ margin: 0; font-size: 26px; font-weight: 600; }}
  h2 {{ color: {text}; font-size: 18px; margin: 20px 0 10px 0; }}
  .meta {{ color: {dim}; font-size: 13px; margin-top: 4px; }}
  .meta code {{ background: {card}; padding: 2px 6px; border-radius: 3px; color: {text}; }}

  .kpi-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
  .kpi-card {{
    background: {card}; border: 1px solid #2a3441; border-radius: 8px;
    padding: 12px 16px; flex: 1; min-width: 140px;
  }}
  .kpi-label {{ color: {dim}; font-size: 11px; text-transform: uppercase;
                letter-spacing: 0.5px; margin-bottom: 5px; }}
  .kpi-value {{ font-size: 22px; font-weight: 600; line-height: 1; }}
  .kpi-subtext {{ color: {dim}; font-size: 11px; margin-top: 4px; }}

  .chart-box {{
    background: {card}; border: 1px solid #2a3441; border-radius: 8px;
    padding: 8px; margin-bottom: 14px;
  }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }}
  @media (max-width: 1100px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.data-table th {{
    background: #1a2332; color: {dim}; text-align: left;
    padding: 8px 10px; border-bottom: 1px solid #2a3441; font-weight: 500;
    position: sticky; top: 0;
  }}
  table.data-table td {{
    padding: 6px 10px; border-bottom: 1px solid #202830; color: {text};
  }}
  table.data-table tr:hover td {{ background: #1e2838; cursor: pointer; }}
  td.run-id {{ font-family: monospace; color: {neutral}; font-weight: 600; }}
  td.params-cell {{ font-family: monospace; font-size: 11px; color: {dim}; }}
  td.params-cell code {{ color: {text}; background: rgba(52, 152, 219, 0.1);
                          padding: 1px 4px; border-radius: 3px; margin-right: 3px; }}

  .tabs {{ display: flex; gap: 4px; border-bottom: 1px solid #2a3441; margin-bottom: 12px; }}
  .tab-btn {{
    background: transparent; border: none; color: {dim};
    padding: 10px 18px; cursor: pointer; font-size: 14px; font-family: inherit;
    border-bottom: 2px solid transparent; transition: all 0.15s;
  }}
  .tab-btn:hover {{ color: {text}; }}
  .tab-btn.active {{ color: {text}; border-bottom-color: #3498db; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  .empty-state {{ color: {dim}; padding: 20px; text-align: center; font-style: italic; }}
  .footer {{ color: {dim}; text-align: center; padding: 20px; font-size: 12px;
             border-top: 1px solid #2a3441; margin-top: 24px; }}

  .table-scroll {{ max-height: 600px; overflow-y: auto;
                    border: 1px solid #2a3441; border-radius: 8px; }}

  details {{ margin: 8px 0; }}
  details summary {{
    cursor: pointer; padding: 10px 14px; background: {card};
    border-radius: 6px; font-weight: 500;
  }}
</style>
</head>
<body>

<header>
  <h1>🎲 Monte Carlo Dashboard</h1>
  <div class="meta">
    Batch <code>{batch_id}</code> · Strategy: <code>{strategy}</code> ·
    {n_completed} runs completed · {period}
  </div>
  <div class="meta" style="margin-top: 4px;">Notes: {notes}</div>
</header>

<section>
  {batch_summary}
</section>

<section>
  <h2>Distribution of All Runs</h2>
  <div class="chart-box">{scatter_chart}</div>
  <div class="two-col">
    <div class="chart-box">{cagr_histogram}</div>
    <div class="chart-box">{param_sensitivity}</div>
  </div>
</section>

<section>
  <h2>Top Runs — Equity Curves</h2>
  <div class="chart-box">{equity_overlay}</div>
</section>

<section>
  <h2>Period Analysis — When and How Top Runs Generate Alpha</h2>
  {period_summary}
  {period_regime_legend}
  <div class="tabs">
    <button class="tab-btn active" data-tab="period_rolling">Rolling 1Y CAGR</button>
    <button class="tab-btn" data-tab="period_excess">Excess vs SPY</button>
    <button class="tab-btn" data-tab="period_yearly">Yearly Heatmap</button>
    <button class="tab-btn" data-tab="period_dd">Drawdowns</button>
    <button class="tab-btn" data-tab="period_corr">Cross-Correlation</button>
  </div>

  <div class="tab-content active" data-tab-content="period_rolling">
    <div class="chart-box">{period_rolling_cagr}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> each line is the trailing 1-year return of one top run.
      A flat line near 20% means stable returns. Wild swings between -30% and +60%
      mean the CAGR was driven by a few lucky periods.
    </p>
  </div>
  <div class="tab-content" data-tab-content="period_excess">
    <div class="chart-box">{period_excess}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> green areas = strategy beating SPY in that 3-month window;
      red = SPY beating strategy. Composite of top 10 runs.
    </p>
  </div>
  <div class="tab-content" data-tab-content="period_yearly">
    <div class="chart-box">{period_yearly}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> column = one top run (sorted left-to-right by CAGR), row = year.
      SPY is in the rightmost column for reference. Spot the years that drove the returns.
    </p>
  </div>
  <div class="tab-content" data-tab-content="period_dd">
    <div class="chart-box">{period_drawdown}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> underwater = how far below the all-time high.
      Long horizontal stretches = long recovery times. Compare with SPY's drawdowns.
    </p>
  </div>
  <div class="tab-content" data-tab-content="period_corr">
    <div class="chart-box">{period_correlation}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> daily-return correlation. All ~1.0 = top runs are basically identical.
      Diverse correlations (0.5-0.8) = real distinct strategies, ensemble might add value.
    </p>
  </div>
</section>

<section>
  <h2>Regime Analysis — How Top Runs Perform in Each Market State</h2>
  {regime_summary}
  {regime_legend}
  <div class="tabs">
    <button class="tab-btn active" data-tab="regime_bars">Return by Regime</button>
    <button class="tab-btn" data-tab="regime_heatmap">Per-Run × Regime</button>
    <button class="tab-btn" data-tab="regime_timeline">Timeline</button>
    <button class="tab-btn" data-tab="regime_duration">Duration</button>
  </div>

  <div class="tab-content active" data-tab-content="regime_bars">
    <div class="chart-box">{regime_avg_bars}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> annualized return of the top 25 runs (median,
      error bars = interquartile range) compared with SPY in each regime.
      If strategy dominates SPY in Bull-Calm but loses in Crisis — the alpha
      is concentrated in calm periods.
    </p>
  </div>
  <div class="tab-content" data-tab-content="regime_heatmap">
    <div class="chart-box">{regime_heatmap}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> each row = one top run, each column = regime.
      Cell shows annualized return in that regime. Green = alpha, red = loss.
      A column of uniform color means all runs behave similarly in that regime.
    </p>
  </div>
  <div class="tab-content" data-tab-content="regime_timeline">
    <div class="chart-box">{regime_timeline}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> each colored segment marks a period of that regime.
      Hover a segment to see exact dates and duration. Cross-reference with the
      Period Analysis charts above — the backgrounds share the same regime colors.
    </p>
  </div>
  <div class="tab-content" data-tab-content="regime_duration">
    <div class="chart-box">{regime_duration}</div>
    <p style="color: #8b949e; font-size: 12px; padding: 0 14px;">
      <strong>How to read:</strong> portion of the 20-year period spent in each regime.
      Context for interpreting regime-specific returns — a regime with only 5%
      of days has less reliable statistics.
    </p>
  </div>
</section>

<section>
  <h2>Run Details</h2>
  <div class="tabs">
    <button class="tab-btn active" data-tab="details_top">Top 25</button>
    <button class="tab-btn" data-tab="details_bottom">Bottom 10</button>
    <button class="tab-btn" data-tab="details_all">All Runs</button>
  </div>

  <div class="tab-content active" data-tab-content="details_top">
    <div class="table-scroll">{top_table}</div>
  </div>
  <div class="tab-content" data-tab-content="details_bottom">
    <div class="table-scroll">{bottom_table}</div>
  </div>
  <div class="tab-content" data-tab-content="details_all">
    <div class="table-scroll">{all_table}</div>
  </div>
</section>

<div class="footer">
  Generated {generated_at} · mc_dashboard.py v1.0
</div>

<script>
  // Each tab strip toggles only siblings within its containing <section>.
  document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const tab = btn.dataset.tab;
      const section = btn.closest('section');
      if (!section) return;
      section.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      section.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const target = section.querySelector(`[data-tab-content="${{tab}}"]`);
      if (target) target.classList.add('active');
    }});
  }});

  // Click on scatter point → show run details
  const scatterDiv = document.getElementById('scatter-runs');
  if (scatterDiv) {{
    scatterDiv.on('plotly_click', function(data) {{
      const runId = data.points[0].customdata;
      alert('Run #' + runId + '\\n\\nTo see full detail, run:\\n  python mc_cli.py show --run-id ' + runId);
    }});
  }}
</script>

</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Main entry
# ══════════════════════════════════════════════════════════════════════════════

def build_mc_dashboard(db: MCDatabase, batch_id: str,
                        benchmarks_path: Optional[str] = None,
                        output_path: Optional[str | Path] = None,
                        top_n_overlay: int = 10) -> Path:
    """
    Build interactive HTML dashboard for a Monte Carlo batch.

    Args:
        db:                MCDatabase instance
        batch_id:          Which batch to visualize
        benchmarks_path:   Optional — for SPY/QQQ overlay
        output_path:       HTML output path
        top_n_overlay:     How many top runs to show equity curves for (max 10 for readability)

    Returns:
        Path to generated HTML
    """
    batch = db.get_batch(batch_id)
    if batch is None:
        raise ValueError(f"Batch {batch_id} not found")

    # Fetch all runs for this batch
    runs_df = db.query_runs(batch_id=batch_id, order_by="cagr_pct", descending=True)
    logger.info(f"Building dashboard for batch {batch_id}: {len(runs_df)} runs")

    # Top/bottom
    top_ids = runs_df.head(25)["run_id"].tolist() if not runs_df.empty else []
    bottom_ids = runs_df.tail(10)["run_id"].tolist() if not runs_df.empty else []
    overlay_ids = top_ids[:top_n_overlay]

    # Build components
    summary_html = _build_batch_summary(batch, runs_df)
    scatter_html = build_scatter_runs(runs_df)
    hist_html = build_cagr_histogram(runs_df)
    sens_html = build_param_sensitivity(runs_df)
    overlay_html = build_equity_overlay(db, overlay_ids, benchmarks_path)

    top_table_html = _build_top_table(runs_df, n=25)
    bottom_table_html = _build_top_table(
        runs_df.tail(10).iloc[::-1].reset_index(drop=True), n=10
    )
    all_table_html = _build_top_table(runs_df, n=500)

    # Period analysis section (5 charts on top runs with detail data)
    try:
        from mc_period_analysis import build_period_analysis_section
        period = build_period_analysis_section(
            db=db, batch_id=batch_id,
            benchmarks_path=benchmarks_path, top_n=25,
        )
    except Exception as e:
        logger.warning(f"Period analysis failed: {e}", exc_info=True)
        empty = '<div class="empty-state">Period analysis unavailable</div>'
        period = {
            "summary": "",
            "rolling_cagr": empty, "excess_return": empty,
            "yearly_heatmap": empty, "drawdown": empty, "correlation": empty,
            "regime_legend": "",
        }

    # Regime analysis section (4 charts based on SPY+VIX regime classification)
    try:
        from mc_regime_analysis import build_regime_section
        regime = build_regime_section(
            db=db, batch_id=batch_id,
            benchmarks_path=benchmarks_path, top_n=25,
        )
    except Exception as e:
        logger.warning(f"Regime analysis failed: {e}", exc_info=True)
        empty = '<div class="empty-state">Regime analysis unavailable</div>'
        regime = {
            "regime_summary": "", "regime_legend": "",
            "regime_timeline": empty, "regime_avg_bars": empty,
            "regime_heatmap": empty, "regime_duration": empty,
        }

    # Period string for header
    if not runs_df.empty:
        ps = runs_df["period_start"].iloc[0] or "?"
        pe = runs_df["period_end"].iloc[0] or "?"
        period_str = f"{ps} → {pe}"
    else:
        period_str = "no data"

    # Render
    html = HTML_TEMPLATE.format(
        bg=BG, card=CARD, text=TEXT, dim=DIM, neutral=NEUTRAL,
        batch_id=batch_id,
        strategy=batch.get("strategy_name", "?"),
        n_completed=len(runs_df),
        period=period_str,
        notes=batch.get("notes") or "—",
        batch_summary=summary_html,
        scatter_chart=scatter_html,
        cagr_histogram=hist_html,
        param_sensitivity=sens_html,
        equity_overlay=overlay_html,
        top_table=top_table_html,
        bottom_table=bottom_table_html,
        all_table=all_table_html,
        period_summary=period["summary"],
        period_regime_legend=period.get("regime_legend", ""),
        period_rolling_cagr=period["rolling_cagr"],
        period_excess=period["excess_return"],
        period_yearly=period["yearly_heatmap"],
        period_drawdown=period["drawdown"],
        period_correlation=period["correlation"],
        regime_summary=regime["regime_summary"],
        regime_legend=regime["regime_legend"],
        regime_avg_bars=regime["regime_avg_bars"],
        regime_heatmap=regime["regime_heatmap"],
        regime_timeline=regime["regime_timeline"],
        regime_duration=regime["regime_duration"],
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    output_path = Path(output_path) if output_path else Path(f"reports/dashboards/mc_{batch_id}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Dashboard saved: {output_path} ({output_path.stat().st_size/1024:.0f} KB)")
    return output_path
