"""Why is this stock in the screen? UI panel component."""
from __future__ import annotations

import html
import logging
import pandas as pd
import streamlit as st

from screener import explain, i18n, theme
from screener.ui.common import L, hebrew_note, t

log = logging.getLogger("screener.app")


def render_explain_card(ticker: str, stock_row: pd.Series, active_filters: dict) -> None:
    """Render a clean card explaining why a stock row matches current active filters."""
    try:
        with st.container(border=True):
            st.subheader(f"מדוע {ticker} מופיעה בסריקה?")

            desc = explain.describe(stock_row)
            if desc:
                st.info(desc)

            items = explain.explain(stock_row, active_filters or {}, lang=i18n.HE)
            if not items:
                st.write("אין פילטרים פעילים — מוצגות כל המניות")
            else:
                lines = ["<ul style='padding-right: 1.25rem; margin-top: 0.5rem;'>"]
                for item in items:
                    mark = "✓" if item["passes"] else "✗"
                    color = theme.GAIN if item["passes"] else theme.LOSS
                    flabel = html.escape(str(item["filter"]))
                    olabel = html.escape(str(item["option"]))
                    val = html.escape(str(item["value"]))
                    lines.append(
                        f"<li style='margin-bottom: 0.25rem;'>"
                        f"<span style='color:{color}; font-weight: bold;'>{mark}</span> "
                        f"<b>{flabel}</b> ({olabel}): {val}"
                        f"</li>"
                    )
                lines.append("</ul>")
                st.html("".join(lines))
    except Exception:
        log.exception("Error rendering explain card")
        hebrew_note("שגיאה בהצגת כרטיס הסבר המניה")


def render(row: pd.Series, selections: dict[str, str]) -> None:
    """Render explanation panel for a stock row given active filter selections."""
    try:
        desc = explain.describe(row)
        if desc:
            st.caption(desc)

        with st.expander(t("Why is this stock in the screen?"), expanded=False, key="ex_expander"):
            items = explain.explain(row, selections, lang=L())
            if not items:
                st.write(t("No active filters — all stocks shown"))
            else:
                lines = []
                for item in items:
                    mark = "✓" if item["passes"] else "✗"
                    color = theme.GAIN if item["passes"] else theme.LOSS
                    flabel = html.escape(str(item["filter"]))
                    olabel = html.escape(str(item["option"]))
                    val = html.escape(str(item["value"]))
                    lines.append(
                        f'<div style="margin-bottom:0.25rem;">'
                        f'<span style="color:{color};font-weight:bold;">{mark}</span> '
                        f'<b>{flabel}</b> ({olabel}): {val}'
                        f'</div>'
                    )
                st.html("".join(lines))
    except Exception:
        log.exception("Error rendering explain panel")
        hebrew_note("שגיאה בהצגת הסבר המניה")
