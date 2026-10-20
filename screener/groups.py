"""
Groups and market heatmap: sector / industry aggregates of the current universe.

group_performance()  one row per group: stock count, total market cap, average Change / Perf W..Y,
                     average Rel Volume and the share of stocks above SMA50. Averages are equal-weight
                     by default (every stock counts once) or market-cap weighted.
heatmap_figure()     Finviz-style treemap: Sector > Industry > Ticker, tile size = market cap,
                     color = daily change (or another % column); groups get the cap-weighted mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from screener.theme import FLAT, GAIN, LOSS

PERF = ["change", "perf_w", "perf_m", "perf_q", "perf_h", "perf_ytd", "perf_y"]
OTHER = "Other"
SEP = "\x1f"     # treemap id separator — cannot appear in names (industries contain "/", e.g. "Investment Banks/Brokers")


def _wmean(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna() & w.notna() & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def group_performance(df: pd.DataFrame, by: str = "sector", weight: str = "median") -> pd.DataFrame:
    """One row per group, largest market cap first. An empty input gives an empty table (same columns)."""
    groups = df[by].fillna(OTHER).astype(str)
    above = (df["price"] > df["sma50"]).astype(float).where(df["sma50"].notna())
    out = pd.DataFrame({"stocks": groups.value_counts(sort=False),
                        "market_cap": df["market_cap"].groupby(groups).sum(min_count=1)})
    w = df["market_cap"].where(df["market_cap"] > 0)
    for c in PERF + ["rel_volume"]:
        if weight == "cap":     # sum(x * cap) / sum(cap), over the stocks that have both
            ok = df[c].notna() & w.notna()
            out[c] = (df[c] * w).where(ok).groupby(groups).sum(min_count=1) / w.where(ok).groupby(groups).sum(min_count=1)
        elif weight == "equal":
            out[c] = df[c].groupby(groups).mean()
        else:                   # median
            out[c] = df[c].groupby(groups).median()
    out["pct_above_sma50"] = above.groupby(groups).mean() * 100
    out = out.rename_axis("group").reset_index()
    return out.sort_values("market_cap", ascending=False, na_position="last").reset_index(drop=True)


def heatmap_data(df: pd.DataFrame, color: str = "change", root: str = "US stocks",
                 stocks: str = "stocks") -> pd.DataFrame:
    """Treemap nodes (id, parent, label, value, color, text): root → sectors → industries → tickers."""
    d = df[(df["market_cap"] > 0)].copy()
    d["sector"] = d["sector"].fillna(OTHER).astype(str)
    d["industry"] = d["industry"].fillna(OTHER).astype(str)
    leaves = pd.DataFrame({
        "id": d["sector"] + SEP + d["industry"] + SEP + d["ticker"], "parent": d["sector"] + SEP + d["industry"],
        "label": d["ticker"], "value": d["market_cap"], "color": d[color], "text": d["company"].fillna(""),
    })
    nodes = [leaves]
    for level, parent in (([ "sector", "industry"], lambda k: k[0]), (["sector"], lambda k: "All")):
        rows = []
        for key, grp in d.groupby(level, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"id": SEP.join(key), "parent": parent(key), "label": key[-1],
                         "value": grp["market_cap"].sum(), "color": _wmean(grp[color], grp["market_cap"]),
                         "text": f"{len(grp)} {stocks}"})
        nodes.append(pd.DataFrame(rows, columns=leaves.columns))
    top = pd.DataFrame([{"id": "All", "parent": "", "label": root, "value": d["market_cap"].sum(),
                         "color": _wmean(d[color], d["market_cap"]), "text": f"{len(d)} {stocks}"}])
    return pd.concat(nodes + [top], ignore_index=True)


def heatmap_figure(df: pd.DataFrame, color: str = "change", limit: float = 3.0, height: int = 720,
                   root: str = "US stocks", stocks: str = "stocks", cap_label: str = "Market cap") -> go.Figure:
    """root / stocks / cap_label are display texts (Hebrew UI passes translations)."""
    n = heatmap_data(df, color, root, stocks)
    val = n["color"].to_numpy(dtype=float)
    has_val = ~np.isnan(val)

    shown_colors = np.where(has_val, np.clip(val, -limit, limit), 0.0)
    marker_colors = list(shown_colors)
    for i, ok in enumerate(has_val):
        if not ok:
            marker_colors[i] = "#808080"

    fmt_color = np.where(has_val, [f"{v:+.2f}%" for v in val], "—")

    fig = go.Figure(go.Treemap(
        ids=n["id"], parents=n["parent"], labels=n["label"], values=n["value"], branchvalues="total",
        customdata=np.column_stack([fmt_color, n["text"]]),
        marker=dict(colors=marker_colors, colorscale=[[0, LOSS], [0.5, FLAT], [1, GAIN]],
                    cmin=-limit, cmax=limit, cmid=0, showscale=True,
                    colorbar=dict(title="%", ticksuffix="%", thickness=12)),
        texttemplate="<b>%{label}</b><br>%{customdata[0]}",
        hovertemplate="<b>%{label}</b> %{customdata[1]}<br>%{customdata[0]}"
                      f"<br>{cap_label} $%{{value:,.3s}}<extra></extra>",
        textfont=dict(color="white"), pathbar=dict(visible=True), maxdepth=4,   # root > sector > industry > ticker
    ))
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=height)
    return fig
