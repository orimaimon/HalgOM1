"""
Market breadth panel UI component.
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from screener import breadth
from screener import i18n as I
from screener.data.store import Store
from screener.theme import ACCENT, MUTED
from screener.ui.common import col_label, hebrew_note, L, ss, t

log = logging.getLogger("screener.app")


def _snapshot_key(store: Store, days: int = 60) -> tuple:
    dates = store.snapshot_dates()
    if days > 0:
        dates = dates[-days:]
    items = []
    for d in dates:
        p = store.snapshot_path(d)
        try:
            mtime = p.stat().st_mtime_ns
        except Exception:
            mtime = 0
        items.append((d.isoformat(), mtime))
    return tuple(items)


@st.cache_data(max_entries=3)
def _get_history_cached(root_str: str, key_tuple: tuple, days: int, junk: bool) -> pd.DataFrame:
    store = Store(Path(root_str))
    return breadth.history(store, days=days, junk=junk)


def _fmt_pct(val) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.1f}%"


def _fmt_int(val) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{int(val):,}"


def render(universe: pd.DataFrame, store: Store, junk: bool = True) -> None:
    try:
        hebrew_note(
            "רוחב שוק מודד כמה מניות משתתפות במגמה: ככל שיותר מניות עולות ונמצאות מעל ממוצעים נעים, "
            "כך המגמה בריאה וחזקה יותר."
        )

        # Today metrics
        b = breadth.breadth(universe, junk=junk)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(breadth.LABELS["pct_above_sma200"], _fmt_pct(b.get("pct_above_sma200")))
        with c2:
            st.metric(breadth.LABELS["pct_above_sma50"], _fmt_pct(b.get("pct_above_sma50")))
        with c3:
            adv_dec_label = f"{breadth.LABELS['advancers']} / {breadth.LABELS['decliners']}"
            adv_dec_val = f"{_fmt_int(b.get('advancers'))} / {_fmt_int(b.get('decliners'))}"
            st.metric(adv_dec_label, adv_dec_val)
        with c4:
            net_val = b.get("net_highs")
            net_str = f"{net_val:+,}" if net_val is not None and not pd.isna(net_val) else "—"
            st.metric(breadth.LABELS["net_highs"], net_str)

        # By sector
        st.subheader(t("By sector"))
        sec_df = breadth.by_sector(universe, junk=junk)
        if not sec_df.empty:
            sec_disp = sec_df.copy()
            sec_disp.index = [I.value("sector", str(s), L()) for s in sec_disp.index]
            sec_disp.index.name = t("Sector")
            st.dataframe(
                sec_disp,
                column_config={
                    "stocks": st.column_config.NumberColumn(t("Stocks"), format="%d"),
                    "pct_above_sma200": st.column_config.NumberColumn(breadth.LABELS["pct_above_sma200"], format="%.1f%%"),
                    "pct_advancing": st.column_config.NumberColumn(breadth.LABELS["pct_advancing"], format="%.1f%%"),
                    "new_highs": st.column_config.NumberColumn(breadth.LABELS["new_highs"], format="%d"),
                    "new_lows": st.column_config.NumberColumn(breadth.LABELS["new_lows"], format="%d"),
                },
                use_container_width=True,
                key="br_sector_table",
            )
        else:
            st.caption(t("No sector data"))

        # Over time
        st.subheader(t("Over time"))
        key_tuple = _snapshot_key(store, days=60)
        hist_df = _get_history_cached(str(store.root), (key_tuple, junk), 60, junk)
        if len(hist_df) < 2:
            hebrew_note("צריך כמה ימים של נתונים שמורים כדי להציג מגמה")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            dates_str = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in hist_df.index]
            fig.add_trace(
                go.Scatter(
                    x=dates_str,
                    y=hist_df["pct_above_sma200"],
                    name=breadth.LABELS["pct_above_sma200"],
                    line=dict(color=ACCENT, width=2),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=dates_str,
                    y=hist_df["net_highs"],
                    name=breadth.LABELS["net_highs"],
                    line=dict(color=MUTED, width=2, dash="dot"),
                ),
                secondary_y=True,
            )
            fig.update_layout(
                margin=dict(t=30, l=10, r=10, b=10),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            fig.update_yaxes(title_text="%", secondary_y=False)
            fig.update_yaxes(title_text=breadth.LABELS["net_highs"], secondary_y=True)
            st.plotly_chart(fig, use_container_width=True, key="br_history_chart")

    except Exception:
        log.exception("error rendering breadth panel")
        hebrew_note("שגיאה בטעינת נתוני רוחב השוק")
