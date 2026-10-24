"""Alerts manager UI component.

Renders active alerts, fired alerts today, and a form to add/remove price, change,
SMA200, and earnings alerts.
"""
from __future__ import annotations

import datetime as dt
import logging
import pandas as pd
import streamlit as st

from screener import alerts, theme
from screener.ui.common import L, col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def _format_threshold(kind: str, value: float | None) -> str:
    if kind in ("cross_above_sma200", "cross_below_sma200") or value is None or pd.isna(value):
        return "—"
    if kind in ("price_below", "price_above"):
        return f"${value:.2f}"
    if kind in ("change_below", "change_above"):
        return f"{abs(value):g}%"
    if kind == "earnings_within":
        return f"{int(value)} ימים"
    return str(value)


def _format_last_fired(last_fired: str | None) -> str:
    if not last_fired:
        return "—"
    try:
        # e.g. "2025-03-14" -> "14/03"
        d = dt.date.fromisoformat(str(last_fired)[:10])
        return d.strftime("%d/%m")
    except Exception:
        return "—"


def fired_today(universe: pd.DataFrame, session_day: dt.date) -> list[alerts.Fired]:
    """Evaluate alerts against the universe for session_day without UI or marking as fired."""
    try:
        loaded = alerts.load()
        if not loaded:
            return []
        return alerts.evaluate(loaded, universe, session_day)
    except Exception:
        log.exception("error evaluating fired_today")
        return []


def _render_body(universe: pd.DataFrame, session_day: dt.date) -> None:
    # 1. Fired today
    all_alerts = alerts.load()
    fired = alerts.evaluate(all_alerts, universe, session_day)
    if fired:
        for f in fired:
            st.info(f.message)
        alerts.mark_fired([f.alert_id for f in fired], session_day)
        all_alerts = alerts.load()

    # 2. Table of existing alerts
    if not all_alerts:
        hebrew_note(t("No alerts yet"))
    else:
        # Header row
        cols = st.columns([1.2, 2.2, 1.5, 2.5, 1.5, 1.0], vertical_alignment="center")
        cols[0].write(f"**{t('Ticker')}**")
        cols[1].write(f"**{t('Alert type')}**")
        cols[2].write(f"**{t('Threshold')}**")
        cols[3].write(f"**{t('Notes')}**")
        cols[4].write(f"**{t('Last fired')}**")
        cols[5].write("")

        # Data rows
        for alert_id, a in all_alerts.items():
            r = st.columns([1.2, 2.2, 1.5, 2.5, 1.5, 1.0], vertical_alignment="center")
            r[0].write(a.ticker)
            r[1].write(alerts.KINDS.get(a.kind, a.kind))
            r[2].write(_format_threshold(a.kind, a.value))
            r[3].write(a.note if a.note and a.note.strip() else "—")
            r[4].write(_format_last_fired(a.last_fired))
            if r[5].button(t("Remove"), key=f"al_rm_{alert_id}"):
                alerts.remove(alert_id)
                st.rerun()

    st.write("")

    # 3. Add alert form
    with st.form(key="al_form"):
        st.write(f"**{t('Add alert')}**")
        c1, c2, c3, c4 = st.columns([1.5, 2.5, 1.5, 2.5], vertical_alignment="bottom")
        ticker_input = c1.text_input(t("Ticker"), key="al_ticker")
        kind = c2.selectbox(
            t("Alert type"),
            options=list(alerts.KINDS.keys()),
            format_func=lambda k: alerts.KINDS[k],
            key="al_kind",
        )
        if kind in ("cross_above_sma200", "cross_below_sma200"):
            val = None
            c3.write("")
        else:
            val = c3.number_input(t("Threshold"), key="al_value", value=0.0)

        note_input = c4.text_input(t("Notes"), key="al_note")
        submitted = st.form_submit_button(t("Add alert"), key="al_submit")

    if submitted:
        ticker = ticker_input.strip().upper()
        valid_tickers = (
            set(universe["ticker"].dropna().unique())
            if (not universe.empty and "ticker" in universe.columns)
            else set()
        )
        if not ticker or ticker not in valid_tickers:
            st.error(f"הסימול {ticker or '—'} לא נמצא בסקרינר")
        else:
            new_alert = alerts.Alert(
                ticker=ticker,
                kind=kind,
                value=float(val) if val is not None else None,
                note=note_input.strip(),
            )
            alerts.add(new_alert)
            st.toast("ההתראה נשמרה")
            st.rerun()


def render(universe: pd.DataFrame, session_day: dt.date) -> None:
    """Render the alerts manager UI component."""
    try:
        _render_body(universe, session_day)
    except Exception:
        log.exception("error rendering alerts view")
        hebrew_note("אירעה שגיאה בטעינת ההתראות")
