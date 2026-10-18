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

PERF = ["change", "perf_w", "perf_m", "perf_q", "perf_h", "perf_ytd", "perf_y"]
OTHER = "Other"


def _wmean(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna() & w.notna() & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def group_performance(df: pd.DataFrame, by: str = "sector", weight: str = "equal") -> pd.DataFrame:
    d = df.assign(_g=df[by].fillna(OTHER).astype(str), _above=(df["price"] > df["sma50"]).astype(float).where(df["sma50"].notna()))
    g = d.groupby("_g", sort=False)
    out = pd.DataFrame({"stocks": g.size(), "market_cap": g["market_cap"].sum(min_count=1)})
    for c in PERF + ["rel_volume"]:
        if weight == "cap":
            out[c] = g[[c, "market_cap"]].apply(lambda x, c=c: _wmean(x[c], x["market_cap"]))
        else:
            out[c] = g[c].mean()
    out["pct_above_sma50"] = g["_above"].mean() * 100
    out = out.rename_axis("group").reset_index()
    return out.sort_values("market_cap", ascending=False, na_position="last").reset_index(drop=True)


def heatmap_data(df: pd.DataFrame, color: str = "change") -> pd.DataFrame:
    """Treemap nodes (id, parent, label, value, color, text): root → sectors → industries → tickers."""
    d = df[(df["market_cap"] > 0)].copy()
    d["sector"] = d["sector"].fillna(OTHER).astype(str)
    d["industry"] = d["industry"].fillna(OTHER).astype(str)
    leaves = pd.DataFrame({
        "id": d["sector"] + "/" + d["industry"] + "/" + d["ticker"], "parent": d["sector"] + "/" + d["industry"],
        "label": d["ticker"], "value": d["market_cap"], "color": d[color], "text": d["company"].fillna(""),
    })
    nodes = [leaves]
    for level, parent in (([ "sector", "industry"], lambda k: k[0]), (["sector"], lambda k: "All")):
        rows = []
        for key, grp in d.groupby(level, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"id": "/".join(key), "parent": parent(key), "label": key[-1],
                         "value": grp["market_cap"].sum(), "color": _wmean(grp[color], grp["market_cap"]),
                         "text": f"{len(grp)} stocks"})
        nodes.append(pd.DataFrame(rows, columns=leaves.columns))
    root = pd.DataFrame([{"id": "All", "parent": "", "label": "US stocks", "value": d["market_cap"].sum(),
                          "color": _wmean(d[color], d["market_cap"]), "text": f"{len(d)} stocks"}])
    return pd.concat(nodes + [root], ignore_index=True)


def heatmap_figure(df: pd.DataFrame, color: str = "change", limit: float = 3.0, height: int = 720) -> go.Figure:
    n = heatmap_data(df, color)
    shown = n["color"].clip(-limit, limit).fillna(0)
    fig = go.Figure(go.Treemap(
        ids=n["id"], parents=n["parent"], labels=n["label"], values=n["value"], branchvalues="total",
        customdata=np.column_stack([n["color"].to_numpy(dtype=float), n["text"]]),
        marker=dict(colors=shown, colorscale=[[0, "#cf222e"], [0.5, "#3a3f47"], [1, "#1a7f37"]],
                    cmin=-limit, cmax=limit, cmid=0, showscale=True,
                    colorbar=dict(title="%", ticksuffix="%", thickness=12)),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
        hovertemplate="<b>%{label}</b> %{customdata[1]}<br>%{customdata[0]:+.2f}%"
                      "<br>Market cap $%{value:,.3s}<extra></extra>",
        textfont=dict(color="white"), pathbar=dict(visible=True), maxdepth=4,   # root > sector > industry > ticker
    ))
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=height)
    return fig
