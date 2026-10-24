"""Portfolio history view component: portfolio value over time vs SPY and QQQ benchmarks, with dividends.

Public entry point: render(plist: PList) -> None.
"""
from __future__ import annotations

import json
import logging
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import portfolio_history, theme
from screener.data import prices
from screener.portfolio import PORTFOLIO, PList
from screener.ui.common import col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def _get_state_mtime() -> float:
    if prices.STATE_FILE.exists():
        try:
            return prices.STATE_FILE.stat().st_mtime
        except OSError:
            return 0.0
    return 0.0


@st.cache_data(max_entries=3)
def _cached_value_series(name: str, rows_json: str, state_mtime: float) -> pd.DataFrame:
    rows = json.loads(rows_json)
    plist = PList(name, PORTFOLIO, rows)
    return portfolio_history.value_series(plist)


def render(plist: PList) -> None:
    if not isinstance(plist, PList) or plist.kind != PORTFOLIO:
        return

    try:
        _render_body(plist)
    except Exception as e:
        log.exception("portfolio_history_view failed: %s", e)
        hebrew_note("לא ניתן להציג את היסטוריית התיק כרגע")


def _render_body(plist: PList) -> None:
    btn = st.button(t("Update price history"), key="ph_update")
    if btn:
        updated, failed = portfolio_history.update_prices({plist.name: plist})
        parts = []
        if updated:
            parts.append(f"{t('Updated')} {len(updated)} {t('stocks')}")
        if failed:
            parts.append(f"{t('Not updated:')} {', '.join(failed)}")
        if parts:
            st.toast(" · ".join(parts))

    rows_json = json.dumps(plist.rows, sort_keys=True, default=str)
    state_mtime = _get_state_mtime()
    vs = _cached_value_series(plist.name, rows_json, state_mtime)

    if vs.empty:
        hebrew_note(t("No history yet — click update price history"))
        return

    missing = vs.attrs.get("missing", [])
    if missing:
        hebrew_note(f"{t('No history for:')} {', '.join(missing)}")

    fig = go.Figure()

    # Net invested (thin line, MUTED)
    fig.add_trace(go.Scatter(
        x=vs.index,
        y=vs["net_invested"],
        mode="lines",
        name=t("Net invested"),
        line=dict(color=theme.MUTED, width=1)
    ))

    # Portfolio value
    fig.add_trace(go.Scatter(
        x=vs.index,
        y=vs["value"],
        mode="lines",
        name=t("Portfolio value"),
        line=dict(color=theme.ACCENT, width=2)
    ))

    # Portfolio + dividends (dashed)
    val_plus_div = vs["value"] + vs["dividends"]
    fig.add_trace(go.Scatter(
        x=vs.index,
        y=val_plus_div,
        mode="lines",
        name=t("Portfolio + dividends"),
        line=dict(color=theme.GAIN, width=2, dash="dash")
    ))

    # SPY
    if "spy" in vs.columns and not vs["spy"].isna().all():
        fig.add_trace(go.Scatter(
            x=vs.index,
            y=vs["spy"],
            mode="lines",
            name="SPY",
            line=dict(width=1.5)
        ))

    # QQQ
    if "qqq" in vs.columns and not vs["qqq"].isna().all():
        fig.add_trace(go.Scatter(
            x=vs.index,
            y=vs["qqq"],
            mode="lines",
            name="QQQ",
            line=dict(width=1.5)
        ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(type="date"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary line
    last = vs.iloc[-1]
    invested = float(last["net_invested"])
    if invested > 0:
        pf_val = float(last["value"]) + float(last["dividends"])
        pf_ret = (pf_val / invested - 1.0) * 100
        pf_str = f"{pf_ret:+.1f}%"

        if pd.notna(last.get("spy")):
            spy_ret = (float(last["spy"]) / invested - 1.0) * 100
            spy_str = f"{spy_ret:+.1f}%"
        else:
            spy_str = "—"

        if pd.notna(last.get("qqq")):
            qqq_ret = (float(last["qqq"]) / invested - 1.0) * 100
            qqq_str = f"{qqq_ret:+.1f}%"
        else:
            qqq_str = "—"

        summary_text = f"{t('Portfolio:')} {pf_str} · SPY: {spy_str} · QQQ: {qqq_str}"
        hebrew_note(summary_text)

    # Dividends received
    divs = portfolio_history.dividends_received(plist)
    if not divs.empty:
        with st.expander(t("Dividends received")):
            disp_divs = divs.copy()
            disp_divs["date"] = pd.to_datetime(disp_divs["date"]).dt.strftime("%d/%m/%Y")
            disp_divs["per_share"] = disp_divs["per_share"].map(lambda v: f"${v:,.2f}")
            disp_divs["shares"] = disp_divs["shares"].map(lambda v: f"{v:,.2f}".rstrip("0").rstrip("."))
            disp_divs["amount"] = disp_divs["amount"].map(lambda v: f"${v:,.2f}")
            disp_divs = disp_divs.rename(columns={
                "date": t("Date"),
                "ticker": t("Ticker"),
                "per_share": "למניה $",
                "shares": t("Quantity"),
                "amount": "סכום $",
            })
            st.dataframe(disp_divs, use_container_width=True, hide_index=True)
