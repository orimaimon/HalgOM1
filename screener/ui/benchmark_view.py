"""Portfolio benchmark comparison view component.

Renders portfolio-vs-benchmark statistics metric cards (beta, alpha, correlation,
tracking error, capture ratios) from a precomputed stats dict.

Public entry point: render(stats: dict | None, benchmark_name: str = "SPY") -> None
"""
from __future__ import annotations

import logging
import math
import pandas as pd
import streamlit as st

from screener.ui.common import hebrew_note

log = logging.getLogger("screener.app")


def _fmt_val(v, is_pct: bool = False) -> str:
    """Format float or None/nan as formatted string or '—'."""
    if v is None:
        return "—"
    if isinstance(v, (float, int)):
        if math.isnan(v):
            return "—"
        try:
            val = float(v)
            if is_pct:
                return f"{val:.2f}%"
            return f"{val:.2f}"
        except (ValueError, TypeError):
            return "—"
    return "—"


def render(stats: dict | None, benchmark_name: str = "SPY") -> None:
    """Render portfolio benchmark statistics metric cards.

    Args:
        stats: Precomputed benchmark stats dict containing n_obs, beta,
            alpha_annual, correlation, tracking_error, up_capture, down_capture.
        benchmark_name: Name of the benchmark index (default "SPY").
    """
    try:
        if not stats:
            st.info("אין מספיק נתונים להשוואה")
            return

        n_obs = stats.get("n_obs")
        if n_obs is None or pd.isna(n_obs) or n_obs <= 0:
            st.info("אין מספיק נתונים להשוואה")
            return

        n_obs_int = int(n_obs)
        if n_obs_int < 60:
            st.warning("מדגם קצר (פחות מ-60 תצפיות) — התוצאות עשויות להיות פחות מייצגות.")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("בטא", _fmt_val(stats.get("beta"), is_pct=False))
        col2.metric("אלפא שנתית", _fmt_val(stats.get("alpha_annual"), is_pct=True))
        col3.metric("מתאם", _fmt_val(stats.get("correlation"), is_pct=False))
        col4.metric("שגיאת מעקב", _fmt_val(stats.get("tracking_error"), is_pct=True))
        col5.metric("תפיסת עליות", _fmt_val(stats.get("up_capture"), is_pct=False))
        col6.metric("תפיסת ירידות", _fmt_val(stats.get("down_capture"), is_pct=True))

        st.caption(
            f"בטא 1 מסמלת תנודתיות זהה למדד ההשוואה ({benchmark_name}). "
            "הנתונים המוצגים אינם מהווים ייעוץ השקעות."
        )

    except Exception:
        log.exception("error rendering benchmark comparison view")
        hebrew_note("שגיאה בהצגת נתוני השוואה למדד")
