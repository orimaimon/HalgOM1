"""Upcoming earnings window view component.

Displays metric cards for earnings buckets, dataframe with days to earnings
and check marks for held stocks, and a warning for held stocks reporting within 7 days.

Public entry point: render(df: pd.DataFrame | None, buckets: dict | None = None) -> None
"""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from screener import i18n
from screener.ui.common import L, hebrew_note, t

log = logging.getLogger("screener.app")


def render(df: pd.DataFrame | None, buckets: dict | None = None) -> None:
    """Render upcoming earnings window view.

    Args:
        df: DataFrame with columns ticker, days_to_earnings, held,
            and optionally market_cap, sector.
        buckets: Dictionary with keys today, this_week, next_week, later
            containing lists of tickers.
    """
    try:
        # If df is None or empty, display a Hebrew info note
        if df is None or df.empty:
            hebrew_note("אין דוחות קרובים להצגה")
            return

        st.subheader("חלון דוחות כספיים קרובים")

        # Metric cards counts
        b = buckets or {}
        if b:
            c_today = len(b.get("today") or [])
            c_this_week = len(b.get("this_week") or [])
            c_next_week = len(b.get("next_week") or [])
            c_later = len(b.get("later") or [])
        else:
            days = df["days_to_earnings"] if "days_to_earnings" in df.columns else pd.Series(dtype=float)
            c_today = int((days == 0).sum()) if not days.empty else 0
            c_this_week = int(((days >= 1) & (days <= 7)).sum()) if not days.empty else 0
            c_next_week = int(((days >= 8) & (days <= 14)).sum()) if not days.empty else 0
            c_later = int((days > 14).sum()) if not days.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("היום", str(c_today))
        c2.metric("השבוע", str(c_this_week))
        c3.metric("שבוע הבא", str(c_next_week))
        c4.metric("מאוחר יותר", str(c_later))

        # Check for held tickers reporting within 7 days
        held_7d: list[str] = []
        if "held" in df.columns and "days_to_earnings" in df.columns:
            for _, row in df.iterrows():
                is_held = bool(row["held"]) and pd.notna(row["held"])
                d = row["days_to_earnings"]
                if is_held and pd.notna(d) and 0 <= float(d) <= 7:
                    tkr = str(row["ticker"])
                    if tkr not in held_7d:
                        held_7d.append(tkr)

        if held_7d:
            st.warning(f"⚠️ מניות בתיק שמדווחות ב-7 הימים הקרובים: {', '.join(held_7d)}")

        # Format dataframe for display
        display_df = df.copy()

        if "held" in display_df.columns:
            display_df["held"] = display_df["held"].map(
                lambda v: "✓" if bool(v) and pd.notna(v) else "—"
            )

        if "sector" in display_df.columns:
            display_df["sector"] = display_df["sector"].map(
                lambda s: i18n.value("sector", s, L()) if pd.notna(s) else "—"
            )

        col_map = {
            "ticker": "סימול",
            "days_to_earnings": "ימים לדוח",
            "held": "מוחזק בתיק",
            "market_cap": "שווי שוק",
            "sector": "סקטור",
        }
        display_df = display_df.rename(columns=col_map)

        st.dataframe(display_df, hide_index=True, use_container_width=True, key="ew_table")

    except Exception:
        log.exception("error rendering earnings window view")
        hebrew_note("שגיאה בהצגת חלון הדוחות הקרובים")
