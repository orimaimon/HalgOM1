"""Groups page (sector / industry averages, track record tab) and Heatmap page.

Moved verbatim from screener/app.py; behavior unchanged."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from screener import groups as G, i18n as I
from screener.fields import GROUP_COLUMNS
from screener.filters import in_index

from screener.ui import breadth_panel, sector_rotation_view
from screener.ui.common import L, sign_color, col_label, display_values, get_store, hebrew_note, junk_toggle, ss, t
from screener.ui.track_record import groups_track_record


# ─────────────────────────── Groups / Heatmap pages ────────────────────────

HEAT_COLORS = {"change": 3, "perf_w": 6, "perf_m": 10, "perf_q": 20, "perf_h": 30, "perf_ytd": 40, "perf_y": 50}
HEAT_UNIVERSES = ["S&P 500", "Nasdaq 100", "Screener results", "All stocks (top 1,000)"]


def groups_page(universe: pd.DataFrame, res: pd.DataFrame) -> None:
    tab_groups, tab_rot, tab_track, tab_breadth = st.tabs(
        [t("Averages"), t("Sector rotation"), t("Track record"), t("Market breadth")]
    )
    with tab_groups:
        c1, c2, c3, c4, c5 = st.columns([1.6, 2.2, 2.0, 1.6, 1.2], vertical_alignment="bottom")
        ss.setdefault("g_by", "Sector")
        ss.setdefault("g_weight", "Median")
        c1.segmented_control(t("Group by"), ["Sector", "Industry"], key="g_by", required=True, format_func=t)
        c2.segmented_control(t("Averages"), ["Median", "Equal", "Market cap"], key="g_weight", required=True, format_func=t,
                             help="חציון = הערך האמצעי (עמיד בפני חריגים). שווה = ממוצע שווה-משקל. שווי שוק = מניות גדולות משפיעות יותר (כמו מדד)")
        base = res if ss.get("g_results") else universe
        by = ss.get("g_by").lower()
        if by == "industry":
            sectors = ["All sectors"] + sorted(base["sector"].dropna().astype(str).unique())
            if ss.get("g_sector") not in sectors:
                ss["g_sector"] = "All sectors"
            c3.selectbox(t("Sector"), sectors, key="g_sector", help="הצגת התעשיות של סקטור אחד בלבד",
                         format_func=lambda v: t(v) if v == "All sectors" else I.value("sector", v, L()))
            if ss.get("g_sector") != "All sectors":
                base = base[base["sector"] == ss.get("g_sector")]
        c4.toggle(t("Only screener results"), key="g_results",
                  help="דלוק: הקבוצות מחושבות רק מהמניות שעברו את הפילטרים בדף הסורק")
        junk_toggle(c5)
        if base.empty:
            st.info(t("No stocks to show - filters or search yielded no results"))
        else:
            weight_mode = "cap" if ss.get("g_weight") == "Market cap" else "equal" if ss.get("g_weight") == "Equal" else "median"
            table = G.group_performance(base, by=by, weight=weight_mode)
            st.markdown(f"#### {len(table):,} {t('sectors' if by == 'sector' else 'industries')} · "
                        f"{len(base):,} {t('stocks')}")
            cc = st.column_config

            def gl(k: str) -> dict:
                return {"label": I.group_column(k, GROUP_COLUMNS[k].label, L()), "help": GROUP_COLUMNS[k].help}

            config = {"group": cc.TextColumn(pinned=True, **gl("group")),
                      "stocks": cc.NumberColumn(format="%d", **gl("stocks")),
                      "market_cap": cc.NumberColumn(format="compact", **gl("market_cap")),
                      "rel_volume": cc.NumberColumn(format="%.2f", **gl("rel_volume"))}
            for k in G.PERF + ["pct_above_sma50"]:
                config[k] = cc.NumberColumn(format="%.2f%%", **gl(k))
            if by == "sector":
                table = display_values(table)
            styled = table.style.map(sign_color, subset=G.PERF)
            st.dataframe(styled, hide_index=True, height=min(38 + 35 * len(table), 760), column_config=config,
                         key="groups_table")
            hebrew_note("ביצועים ממוצעים של המניות בכל קבוצה. לחיצה על כותרת עמודה ממיינת. "
                        "מומלץ להשאיר את הסתרת מניות הזבל דלוקה — מניות זבל עם תנועות של מאות אחוזים "
                        "מעוותות ממוצע שווה-משקל.")
    with tab_rot:
        sector_rotation_view.render(universe)
    with tab_track:
        groups_track_record()
    with tab_breadth:
        breadth_panel.render(universe, get_store(), ss.get("hide_junk", True))


def heatmap_page(universe: pd.DataFrame, res: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([2.4, 2.4, 1.2], vertical_alignment="bottom")
    ss.setdefault("h_universe", HEAT_UNIVERSES[0])
    ss.setdefault("h_color", "change")
    c1.selectbox(t("Stocks"), HEAT_UNIVERSES, key="h_universe", format_func=t,
                 help="אילו מניות להציג. S&P 500 מקביל למפת השוק של Finviz")
    c2.selectbox(t("Color by"), list(HEAT_COLORS), key="h_color", format_func=col_label,
                 help="ירוק = עלייה, אדום = ירידה. הצבע רווי בקצוות הסקאלה (למשל ±3% לשינוי יומי)")
    junk_toggle(c3)
    match ss.get("h_universe"):
        case "Screener results":
            data = res
        case "All stocks (top 1,000)":
            data = universe.nlargest(1000, "market_cap")
        case label:
            data = universe[in_index(label)(universe)]
    if data.empty:
        st.info(t("No stocks to show"))
        return
    color = ss.get("h_color")
    fig = G.heatmap_figure(display_values(data), color, limit=HEAT_COLORS[color], root=t("US stocks"),
                           stocks=t("stocks"), cap_label=t("Market cap"))
    st.plotly_chart(fig, key="heatmap", config={"displaylogo": False})
    hebrew_note(f"{len(data):,} מניות. גודל = שווי שוק, צבע = {col_label(color)}. "
                "לחיצה על סקטור או תעשייה מגדילה אותם; לחזרה לחץ על שורת הניווט למעלה.")
