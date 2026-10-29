"""Screen Diff View UI component."""
from __future__ import annotations

import logging
import streamlit as st

from screener.ui.common import hebrew_note

log = logging.getLogger("screener.app")


def render(result: dict | None, label: str = "") -> None:
    """Render screen diff dict showing entered, left, stayed tickers and list size changes."""
    try:
        if label:
            st.subheader(label)

        if not isinstance(result, dict):
            st.info("אין נתוני השוואה זמינים")
            return

        required_keys = {"entered", "left", "stayed", "n_old", "n_new"}
        if not required_keys.issubset(result.keys()):
            st.info("נתוני השוואה חסרים או שגויים")
            return

        entered = result.get("entered")
        left = result.get("left")
        stayed = result.get("stayed")
        n_old = result.get("n_old")
        n_new = result.get("n_new")

        if not isinstance(entered, list) or not isinstance(left, list) or not isinstance(stayed, list):
            st.info("נתוני השוואה חסרים או שגויים")
            return

        if not isinstance(n_old, int) or not isinstance(n_new, int):
            st.info("נתוני השוואה חסרים או שגויים")
            return

        # Metric row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("נכנסו", len(entered))
        with c2:
            st.metric("יצאו", len(left))
        with c3:
            st.metric("נשארו", len(stayed))
        with c4:
            st.metric("גודל קודם", n_old)
        with c5:
            st.metric("גודל חדש", n_new)

        if not entered and not left:
            st.info("אין שינוי")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**נכנסו:**")
                st.write(", ".join(entered) if entered else "—")
            with col2:
                st.markdown("**יצאו:**")
                st.write(", ".join(left) if left else "—")

    except Exception:
        log.exception("Error rendering screen diff view")
        hebrew_note("שגיאה בהצגת נתוני השינוי")
