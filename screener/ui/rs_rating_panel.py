"""
Relative Strength (RS) Rating panel UI component.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from screener import i18n as I
from screener.theme import ACCENT
from screener.ui.common import hebrew_note, L

log = logging.getLogger("screener.app")


def _prep_display_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    disp = df[cols].copy()
    if "sector" in disp.columns:
        disp["sector"] = disp["sector"].map(lambda s: I.value("sector", str(s), L()) if pd.notna(s) else s)
    return disp


def render(df: pd.DataFrame, top_n: int = 15) -> None:
    try:
        if df is None or df.empty or "rs_rating" not in df.columns:
            hebrew_note("אין נתוני דירוג עוצמה יחסית (RS Rating) להצגה.")
            return

        valid_df = df[df["rs_rating"].notna()].copy()
        if valid_df.empty:
            hebrew_note("אין נתוני דירוג עוצמה יחסית (RS Rating) להצגה.")
            return

        st.subheader("דירוג עוצמה יחסית (RS Rating)")

        cols = [c for c in ["ticker", "rs_rating", "price", "sector"] if c in valid_df.columns]

        column_config = {
            "ticker": st.column_config.TextColumn("סימול"),
            "rs_rating": st.column_config.NumberColumn("דירוג RS", format="%d"),
            "price": st.column_config.NumberColumn("מחיר", format="%.2f"),
            "sector": st.column_config.TextColumn("סקטור"),
        }

        strongest_df = valid_df.sort_values(by="rs_rating", ascending=False).head(top_n)
        weakest_df = valid_df.sort_values(by="rs_rating", ascending=True).head(top_n)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### החזקות ביותר")
            st.dataframe(
                _prep_display_df(strongest_df, cols),
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                key="rs_top_table",
            )
        with c2:
            st.markdown("##### החלשות ביותר")
            st.dataframe(
                _prep_display_df(weakest_df, cols),
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                key="rs_bottom_table",
            )

        fig = px.histogram(
            valid_df,
            x="rs_rating",
            nbins=20,
            title="התפלגות דירוגי RS",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(
            xaxis_title="דירוג RS",
            yaxis_title="מספר מניות",
            margin=dict(t=30, l=10, r=10, b=10),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True, key="rs_histogram_chart")

    except Exception:
        log.exception("error rendering rs rating panel")
        hebrew_note("שגיאה בהצגת דירוג עוצמה יחסית")
