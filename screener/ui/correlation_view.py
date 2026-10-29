"""Portfolio correlation heatmap and market stress-test UI component.

Public entry point: render(plist: PList, universe: pd.DataFrame) -> None
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import portfolio_risk, theme
from screener.data import prices
from screener.portfolio import WATCHLIST, PList
from screener.ui.common import col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def _fmt_money(v: float) -> str:
    if pd.isna(v):
        return "—"
    if v < 0:
        return f"-${abs(v):,.0f}"
    return f"${v:,.0f}"


def render(plist: PList, universe: pd.DataFrame) -> None:
    """Renders market stress-test scenarios and 90-day return correlation matrix."""
    try:
        if plist is None or getattr(plist, "kind", None) == WATCHLIST:
            return

        exp = portfolio_risk.exposure(plist, universe)
        if exp.empty:
            return

        # Section 1 — Market stress test
        st.subheader(t("Market stress scenarios"))

        total_val = float(exp["value"].sum()) if "value" in exp.columns else 0.0

        # Compute portfolio weighted beta, defaulting missing stock beta to 1.0
        betas = exp["beta"].fillna(1.0) if "beta" in exp.columns else pd.Series(1.0, index=exp.index)
        values = exp["value"] if "value" in exp.columns else pd.Series(0.0, index=exp.index)

        if total_val > 0:
            weighted_beta = float((values * betas).sum() / total_val)
        else:
            weighted_beta = 1.0

        scenarios = [-5.0, -10.0, -20.0, 5.0, 10.0]
        stress_rows = []

        for sc in scenarios:
            est_impact_pct = sc * weighted_beta
            pnl_dollars = total_val * (est_impact_pct / 100.0)
            est_val = total_val + pnl_dollars

            stress_rows.append({
                t("Market change"): f"{sc:+.1f}%",
                t("Estimated impact %"): f"{est_impact_pct:+.1f}%",
                t("Estimated P&L ($)"): _fmt_money(pnl_dollars),
                t("Estimated value"): _fmt_money(est_val),
            })

        stress_df = pd.DataFrame(stress_rows)
        st.dataframe(
            stress_df,
            hide_index=True,
            use_container_width=True,
            key="cr_stress_table",
        )

        # Section 2 — Return correlation
        st.subheader(t("Return correlation (90 days)"))

        tickers = exp["ticker"].tolist() if "ticker" in exp.columns else []

        # Collect price histories
        price_series_dict = {}
        for ticker in tickers:
            hist = prices.history(ticker)
            if not hist.empty:
                col = "adj_close" if "adj_close" in hist.columns else "close"
                if col in hist.columns:
                    s = hist[col].dropna()
                    if not s.empty:
                        price_series_dict[ticker] = s

        if len(price_series_dict) < 2:
            hebrew_note(t("Not enough price history for correlation matrix"))
            return

        closes_df = pd.DataFrame(price_series_dict).sort_index().tail(90)
        rets_df = closes_df.pct_change()

        corr_df = rets_df.corr(min_periods=20)

        # Check if fewer than 2 positions have at least 20 overlapping days
        valid_pairs = False
        if not corr_df.empty and len(corr_df.columns) >= 2:
            for i in range(len(corr_df.columns)):
                for j in range(i + 1, len(corr_df.columns)):
                    c_val = corr_df.iloc[i, j]
                    if not pd.isna(c_val):
                        valid_pairs = True
                        break
                if valid_pairs:
                    break

        if not valid_pairs:
            hebrew_note(t("Not enough price history for correlation matrix"))
            return

        # High-correlation warnings (corr >= 0.80)
        weights_map = dict(zip(exp["ticker"], exp["weight"])) if "weight" in exp.columns else {}
        cols = corr_df.columns.tolist()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                t1, t2 = cols[i], cols[j]
                c = corr_df.loc[t1, t2]
                if not pd.isna(c) and c >= 0.80:
                    w1 = float(weights_map.get(t1, 0.0))
                    w2 = float(weights_map.get(t2, 0.0))
                    st.warning(f"{t1} ו-{t2} זזות יחד (מתאם {round(c * 100)}%) ומהוות יחד {round(w1 + w2)}% מהתיק")

        # Plotly heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                zmin=-1.0,
                zmax=1.0,
                texttemplate="%{z:.2f}",
                colorscale=[
                    [0.0, theme.LOSS],
                    [0.5, theme.FLAT],
                    [1.0, theme.ACCENT],
                ],
            )
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=max(250, len(corr_df) * 40 + 40),
        )
        st.plotly_chart(fig, use_container_width=True, key="cr_heatmap")

    except Exception:
        log.exception("Error rendering correlation view")
        hebrew_note(t("Error displaying correlation view"))
