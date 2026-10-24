"""Portfolio risk UI component: weights, sector breakdown, concentration warnings, and weighted beta.

Public entry point: render(plist: PList, universe: pd.DataFrame) -> None
"""
from __future__ import annotations

import logging
import pandas as pd
import plotly.express as px
import streamlit as st

from screener import i18n, portfolio_risk, theme
from screener.portfolio import WATCHLIST, PList
from screener.ui.common import L, col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def render(plist: PList, universe: pd.DataFrame) -> None:
    """Renders the portfolio risk view. Watchlists produce no output."""
    try:
        if plist.kind == WATCHLIST:
            return

        exp = portfolio_risk.exposure(plist, universe)
        summary = portfolio_risk.summary(exp)

        if exp.empty or summary["positions"] == 0:
            hebrew_note(t("No open positions"))
            return

        # Metrics row: positions, top 5 weight %, weighted beta with coverage
        c1, c2, c3 = st.columns(3)
        c1.metric(t("Positions"), str(summary["positions"]))
        c2.metric(t("Top 5 weight"), f"{summary['top5_weight']:.1f}%")

        if summary["beta"] is None:
            beta_str = "—"
            delta_str = None
        else:
            beta_str = f"{summary['beta']:.2f}"
            cov = summary.get("beta_coverage")
            delta_str = f"לפי {round(cov)}% מהתיק" if cov is not None and cov < 100 else None

        c3.metric(t("Weighted beta"), beta_str, delta=delta_str, delta_color="off")

        # Warnings
        for w in summary.get("warnings", []):
            st.warning(w)

        # By sector breakdown horizontal bar chart
        sec_df = portfolio_risk.by_sector(exp)
        if not sec_df.empty:
            chart_df = sec_df.copy()
            chart_df["sector_name"] = chart_df["sector"].map(lambda s: i18n.value("sector", s, L()))
            # Sort weight ascending so highest weight is rendered at top in horizontal orientation
            chart_df = chart_df.sort_values("weight", ascending=True)

            fig = px.bar(
                chart_df,
                x="weight",
                y="sector_name",
                orientation="h",
                text_auto=".1f",
            )
            fig.update_traces(
                marker_color=theme.ACCENT,
                textposition="auto",
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                margin=dict(l=0, r=0, t=10, b=0),
                height=max(180, len(chart_df) * 35 + 40),
            )
            st.plotly_chart(fig, use_container_width=True, key="rk_sector_chart")

        # Per-stock weights table
        stock_df = pd.DataFrame()
        stock_df[col_label("ticker")] = exp["ticker"]
        stock_df[col_label("sector")] = exp["sector"].map(lambda s: i18n.value("sector", s, L()))
        stock_df[t("Weight")] = exp["weight"].map(lambda w: f"{w:.1f}%" if pd.notna(w) else "—")
        stock_df[col_label("beta")] = exp["beta"].map(lambda b: f"{b:.2f}" if pd.notna(b) else "—")

        st.dataframe(stock_df, hide_index=True, use_container_width=True, key="rk_stock_table")

    except Exception:
        log.exception("error rendering portfolio risk view")
        hebrew_note(t("Error displaying portfolio risk"))
