"""Screener page: toolbar, filter panel, results table, stock detail and export.

Moved verbatim from screener/app.py; behavior unchanged."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from screener import export, history, i18n as I, portfolio as PF, presets as P, theme
from screener.fields import COLUMNS, DEFAULT_SORT, VIEWS
from screener.filters import ANY, FILTERS, GROUPS, active, option_labels

import datetime as dt
import hashlib
import json

from screener.filters import FILTERS_BY_KEY
from screener.ui import compare_panel, explain_panel, sizing_panel
from screener.ui.common import (CUSTOM, L, USER_PREFIX, sign_color, all_presets, col_label,
    column_config, display_values, hebrew_note,
    junk_toggle, log, on_reset, opt, preset_modified,
    selections, ss, t)
from screener.ui.watchlists import NEW_WATCHLIST, on_add_to_watchlist


GRID = 6


def escape_markdown(text: str) -> str:
    """Escape markdown special characters so company names containing $, *, _, etc. render literally."""
    if text is None or pd.isna(text):
        return "—"
    text = str(text)
    if text == "—" or text.lower() in ("nan", "none"):
        return "—"
    for ch in ("\\", "*", "_", "`", "~", "$", "[", "]"):
        text = text.replace(ch, "\\" + ch)
    return text


def _clean_str(v) -> str:
    if v is None or pd.isna(v) or v == "" or str(v).lower() in ("nan", "none"):
        return "—"
    return str(v)


def rows_key(res: pd.DataFrame) -> str:
    """Table widget key tied to the rows shown: the selection is a row position, so when the rows change
    (filters, Hide junk, Only new, sort) the key changes and the old selection is dropped instead of
    opening the chart of whatever stock now sits in that position."""
    return "table_" + hashlib.md5("\x1f".join(res["ticker"].astype(str)).encode()).hexdigest()[:12]


def selection_index(res: pd.DataFrame, ticker: str | None) -> dict | None:
    if not ticker or res.empty:
        return None
    matches = res.index[res["ticker"] == ticker].tolist()
    if matches:
        return {"selection": {"rows": [res.index.get_loc(matches[0])]}}
    return None


JUNK_CONFLICT_OPTIONS = {
    "market_cap": "Nano (under $50M)",
    "avg_volume": "Under 50K",
    "price": "Under $1",
}


def junk_conflict_hint() -> str | None:
    if not ss.get("hide_junk", True):
        return None
    sel = selections()
    conflicts = []
    for k, v in JUNK_CONFLICT_OPTIONS.items():
        if sel.get(k) == v:
            conflicts.append(f"“{opt(v)}”")
    if not conflicts:
        return None
    c_text = ", ".join(conflicts)                # explanations are Hebrew in both languages
    return f"⚠️ האפשרות {c_text} מתנגשת עם “הסתר זבל” ולא תוחזר אף מניה. כדי לראות תוצאות, יש לכבות את “הסתר זבל”."


def described() -> list[tuple[str, str]]:
    """(filter, option) of the active filters in the display language — for the chips and the exports."""
    return [(I.filter_label(k, FILTERS_BY_KEY[k].label, L()), opt(v)) for k, v in active(selections()).items()]


def find_removed_pill(active_keys: list[str], selected_pills: list[str]) -> str | None:
    """Given the old active filters and the new selected pills, return the key that was deselected."""
    for key in active_keys:
        if key not in selected_pills:
            return key
    return None


def pills_key(labels: list[str]) -> str:
    """A new pills widget for every set of active filters: a kept widget would keep its old selection,
    and a filter added later would show as an unselected (removed-looking) pill."""
    return "active_pills_" + hashlib.md5("|".join(labels).encode()).hexdigest()[:10]


def on_remove_pill(key: str, filter_of: dict[str, str]):
    removed = find_removed_pill(list(filter_of), ss.get(key) or [])
    if removed:
        target = filter_of[removed]
        if target == "query":
            ss["query"] = ""
        else:
            ss[f"f_{target}"] = ANY


HEBREW_WEEKDAYS = {0: "ב'", 1: "ג'", 2: "ד'", 3: "ה'", 4: "ו'", 5: "ש'", 6: "א'"}   # date.weekday(): Monday = 0


def hebrew_day_str(d: dt.date) -> str:
    """'יום ו' 16/10' — the compared trading day, for the NEW note."""
    return f"יום {HEBREW_WEEKDAYS[d.weekday()]} {d:%d/%m}"


def fmt_value(key: str, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NaT or v == "":
        return "—"
    kind = COLUMNS[key].kind
    if kind == "pct":
        return f"{v:+.2f}%" if key in export._SIGNED else f"{v:.2f}%"
    if kind == "price":
        return f"${v:,.2f}"
    if kind in ("money", "count"):
        prefix = "$" if kind == "money" else ""
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(v) >= div:
                return f"{prefix}{v / div:.2f}{suf}"
        return f"{prefix}{v:,.0f}"
    if kind == "date":
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if kind in ("days", "score", "int"):
        return f"{v:,.0f}"
    if kind == "range":
        return f"{v:.0%}"
    if kind == "ratio":
        return f"{v:,.2f}"
    if kind == "flag":
        return ("כן" if v else "לא") if L() == I.HE else ("yes" if v else "no")
    return str(v)


def tradingview_chart(symbol: str, height: int = 560) -> None:
    locale = "he_IL" if L() == I.HE else "en"
    config = {
        "autosize": False, "width": "100%", "height": height - 34, "symbol": symbol, "interval": "D", "timezone": "America/New_York",
        "theme": "light", "style": "1", "locale": locale, "allow_symbol_change": True,
        "hide_side_toolbar": False, "details": False, "calendar": False,
        "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies"],
        "support_host": "https://www.tradingview.com",
    }
    st.iframe(f"""
<style>html, body {{margin: 0; height: 100%;}}</style>
<div class="tradingview-widget-container" style="height:{height - 10}px;width:100%">
  <div class="tradingview-widget-container__widget" style="height:{height - 34}px;width:100%"></div>
  <div class="tradingview-widget-copyright" style="font:11px sans-serif">
    <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">Chart by TradingView</a>
  </div>
  <script type="text/javascript"
    src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {json.dumps(config)}
  </script>
</div>""", height=height)


from screener.ui import chart_flip_view  # noqa: E402 — imported after tradingview_chart to avoid circular import


# ─────────────────────────── callbacks ─────────────────────────────────────

def on_preset():
    p = all_presets().get(ss.get("preset"))
    if p is None:
        return
    for f in FILTERS:
        ss[f"f_{f.key}"] = p.filters.get(f.key, ANY)
    ss["view"] = p.view if p.view in VIEWS else "Overview"
    ss["sort_col"] = p.sort if p.sort in COLUMNS else DEFAULT_SORT[0]
    ss["sort_asc"] = p.ascending
    ss["query"] = ""


def on_clear_all():
    """The empty-results button: every filter, the search and "Only new"."""
    on_reset()


def on_list_banner_close():
    ss["query"] = ""
    ss.pop("list_view", None)


def on_save():
    name = (ss.get("save_name") or "").strip()
    if not name:
        return
    user = P.load_user()
    if name in user and ss.get("confirm_save_name") != name:
        ss["confirm_save_name"] = name
        return
    ss.pop("confirm_save_name", None)
    try:
        P.save_user(name, P.Preset(filters=active(selections()), view=ss.get("view"),
                                   sort=ss.get("sort_col"), ascending=ss.get("sort_asc")))
    except Exception as e:
        log.warning("cannot save screen %r: %s", name, e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    ss["preset"] = USER_PREFIX + name
    ss["save_name"] = ""
    st.toast(f"{t('Saved screen')}: “{name}”", icon="💾")


def on_delete():
    name = ss.get("delete_name")
    if name:
        try:
            P.delete_user(name)
        except Exception as e:
            log.warning("cannot delete screen %r: %s", name, e)
            st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
            return
        if ss.get("preset") == USER_PREFIX + name:
            ss["preset"] = CUSTOM
        st.toast(f"{t('Deleted')}: “{name}”", icon="🗑️")


def on_asset_type_change() -> None:
    on_reset()
    if ss.get("asset_type") == "etfs":
        ss["view"] = "ETF" if "ETF" in VIEWS else "Overview"
        ss["sort_col"] = "aum" if "aum" in COLUMNS else DEFAULT_SORT[0]
        ss["sort_asc"] = False
    else:
        ss["view"] = "Overview"
        ss["sort_col"] = DEFAULT_SORT[0]
        ss["sort_asc"] = DEFAULT_SORT[1]


def toolbar(presets: dict[str, P.Preset]) -> None:
    t_top1, t_top2 = st.columns([2.2, 7.8], vertical_alignment="center")
    t_top1.segmented_control("Asset type", ["stocks", "etfs"], key="asset_type",
                             default="stocks", format_func=lambda x: t("Stocks") if x == "stocks" else t("ETFs"),
                             on_change=on_asset_type_change, label_visibility="collapsed")
    names = [CUSTOM] + list(presets)
    if ss.get("preset") not in names:
        ss["preset"] = CUSTOM
    c1, c2, c3, c4, c7, c5, c6 = st.columns([2.2, 2.0, 1.5, 1.1, 1.6, 1.0, 1.3], vertical_alignment="bottom")
    changed = ss.get("preset") if preset_modified(presets) else None
    lang = L()

    def preset_name(n: str) -> str:
        name = I.ui(n, lang) if n == CUSTOM else n if n.startswith(USER_PREFIX) else I.preset(n, lang)
        return f"{name} {I.ui('(changed)', lang)}" if n == changed else name

    c1.selectbox(t("Screen"), names, key="preset", on_change=on_preset, format_func=preset_name,
                 help="סריקות מוכנות. בחירה בסריקה ממלאת את הפילטרים — אפשר לשנות אותם אחר כך")
    c2.text_input(t("Search"), key="query", placeholder=t("Ticker / company, or list: AAPL, MSFT"),
                  help="חיפוש לפי סימול או שם חברה. אפשר להדביק רשימת סימולים מופרדת בפסיקים")
    sortable = [k for k, c in COLUMNS.items() if c.kind not in ("text", "flag") or k in ("ticker", "company")]
    c3.selectbox(t("Order by"), sortable, key="sort_col", format_func=col_label)
    c4.toggle(t("Ascending"), key="sort_asc")
    junk_toggle(c7)
    c5.button(t("Reset"), on_click=on_reset, use_container_width=True, help="ניקוי כל הפילטרים")
    with c6.popover(t("Save"), icon=":material/bookmark:", use_container_width=True, help="שמירת הסריקה הנוכחית / מחיקת סריקה שמורה"):
        save_val = (ss.get("save_name") or "").strip()
        user = P.load_user()
        st.text_input(t("Save current screen as"), key="save_name")
        if save_val and save_val in user and ss.get("confirm_save_name") == save_val:
            st.warning(t("A screen with this name already exists. Click save again to overwrite."))
        st.button(t("Save"), on_click=on_save, type="primary")
        if user:
            st.divider()
            st.selectbox(t("Delete saved screen"), list(user), key="delete_name")
            st.button(t("Delete"), on_click=on_delete)
    note = presets.get(ss.get("preset")).note if ss.get("preset") in presets else ""
    if note:
        hebrew_note(f"ℹ️ {note}")
    if changed:
        n1, n2 = st.columns([4, 1.3], vertical_alignment="center")
        with n1:
            hebrew_note(t("Filters have been changed since the screen was selected."))
        n2.button(t("Restore screen"), on_click=on_preset, key="preset_reapply", use_container_width=True,
                  help="מחזיר את הפילטרים של הסריקה שנבחרה")


def filter_panel(df: pd.DataFrame) -> None:
    sel = selections()
    n_active = len(active(sel))
    title = f"{t('Filters')} ({n_active} {t('active')})" if n_active else t("Filters")
    with st.expander(title, expanded=n_active > 0):
        hint = junk_conflict_hint()
        if hint:
            hebrew_note(hint)
        is_etf = ss.get("asset_type") == "etfs"
        active_groups = ["Descriptive", "Technical", "Signals"] if is_etf else GROUPS
        counts = {g: sum(1 for f in FILTERS if f.group == g and sel[f.key] != ANY) for g in active_groups}
        tabs = st.tabs([f"{t(g)} ({counts[g]})" if counts[g] else t(g) for g in active_groups])
        for tab, group in zip(tabs, active_groups):
            with tab:
                if is_etf:
                    group_filters = [f for f in FILTERS if f.group == group and (f.column in df.columns and (df[f.column].notna().any() or f.key in ("aum", "expense_ratio")))]
                else:
                    group_filters = [f for f in FILTERS if f.group == group and f.key not in ("aum", "expense_ratio")]
                for i in range(0, len(group_filters), GRID):
                    cols = st.columns(GRID)
                    for col, f in zip(cols, group_filters[i:i + GRID]):
                        labels = option_labels(f, df)
                        key = f"f_{f.key}"
                        if ss.get(key, ANY) not in labels:
                            ss[key] = ANY
                        col.selectbox(I.filter_label(f.key, f.label, L()), labels, key=key, help=f.tooltip,
                                      format_func=opt)


def results(df: pd.DataFrame, universe: pd.DataFrame, res: pd.DataFrame, snap,
            prev: history.Previous | None) -> None:
    top = st.columns([3, 5, 1.6], vertical_alignment="center")
    n_new = int((res["new"] == history.NEW).sum()) if prev is not None else 0
    if prev is not None and ss.get("only_new"):          # the count must match the table below it
        res = res[res["new"] == history.NEW]
    of = f"<small>/ {len(universe):,}</small>" if len(res) != len(universe) else ""
    top[0].markdown(f'<div class="hs-count">{len(res):,} {html.escape(t("results"))}{of}</div>',
                    unsafe_allow_html=True)
    banner = ss.get("list_view")
    if banner and ss.get("query") == banner[1]:
        b1, b2 = st.columns([4, 1.3], vertical_alignment="center")
        with b1:
            hebrew_note(f"{t('Showing list')} “{banner[0]}” {t('from Portfolios page')}")
        b2.button(t("Back to all stocks"), on_click=on_list_banner_close, key="list_banner_close",
                  use_container_width=True)
    with top[1]:
        st.segmented_control("View", list(VIEWS), key="view", label_visibility="collapsed", required=True,
                             format_func=t)
    view_cols = VIEWS[ss.get("view")]
    chips = described()
    items = [(f"{k}: {v}", f) for (k, v), f in zip(chips, active(selections()))]
    query_val = (ss.get("query") or "").strip()
    if query_val:
        items.append((f"{t('Search')}: {query_val}", "query"))
    if items:
        filter_of = {label: target for label, target in items}
        labels = list(filter_of)
        key = pills_key(labels)
        st.pills(t("Active filters"), options=labels, default=labels, selection_mode="multi",
                 label_visibility="collapsed", key=key, on_change=on_remove_pill, args=(key, filter_of),
                 help="לחיצה על פילטר מבטלת אותו")
    notes = []                                   # end-user explanations are Hebrew (CLAUDE.md)
    if ss.get("hide_junk") and len(universe) < len(df):
        notes.append(f"{len(df) - len(universe):,} מניות זבל מוסתרות ({t('Hide junk')})")
    if prev is not None:
        day_str = hebrew_day_str(prev.day)
        top[2].toggle(f"{t('Only new')} ({n_new:,})", key="only_new",
                      help=f"רק מניות שנכנסו לתוצאות מאז ה-snapshot של {day_str}")
        view_cols = [view_cols[0], "new", *view_cols[1:]]
        notes.append(f"חדש = לא היה בתוצאות ב{day_str}")
    if notes:
        hebrew_note(" · ".join(notes))
    if prev is not None and prev.missing:
        day_str = hebrew_day_str(prev.day)
        hebrew_note(f"ההשוואה ל-{day_str} חלקית: ה-snapshot נשמר בגרסה קודמת בלי {len(prev.missing)} "
                    "שדות חדשים, ולכן פילטרים על השדות האלה מסמנים יותר מניות כ-NEW. יסתדר מעצמו ממחר.")

    if res.empty:
        hebrew_note("אף מניה לא עומדת בכל התנאים. אפשר להחליש או להסיר פילטר (הם מופיעים למעלה), "
                    "לנקות את החיפוש, או לאפס הכול בלחיצה אחת:")
        st.button(t("Clear all filters"), on_click=on_clear_all, type="primary", key="clear_all")
        return

    table = display_values(res[view_cols])
    if L() == I.HE and "signals" in table.columns:   # grid only: Excel/CSV get the plain text
        table["signals"] = table["signals"].map(I.rtl_cell)
    signed = [c for c in view_cols if c in export._SIGNED]
    if signed:
        table = table.style.map(sign_color, subset=signed)

    sel_ticker = ss.get("selected_ticker")
    sel_default = selection_index(res, sel_ticker)

    event = st.dataframe(
        table, hide_index=True, height=560, key=rows_key(res),
        column_config={k: column_config(k) for k in view_cols},
        on_select="rerun", selection_mode="single-row", placeholder="—",
        selection_default=sel_default,
    )
    rows = event.selection.rows if event and event.selection else []
    if rows and rows[0] < len(res):
        selected_row = res.iloc[rows[0]]
        ss["selected_ticker"] = selected_row["ticker"]
        detail(selected_row)
    else:
        ss.pop("selected_ticker", None)
        hebrew_note(t("👆 Click a row in the table to open chart and stock details"))
    with st.expander(t("Fast chart review"), expanded=False):
        chart_flip_view.render(res, selections())
    compare_panel.render(universe, list(res["ticker"]))
    export_bar(res, view_cols, snap)


DETAIL_FIELDS = [
    ("Valuation & size", ["market_cap", "pe", "fwd_pe", "peg", "ps", "pb", "ev_ebitda"]),
    ("Growth & quality", ["eps_growth", "eps_growth_q", "sales_growth_q", "roe", "roic", "net_margin", "debt_eq"]),
    ("Technical", ["rsi", "rsi_w", "sma50_pct", "sma150_pct", "sma200_pct", "high52_pct", "range52", "atr_pct",
                   "beta", "rel_volume"]),
    ("Signals", ["n_signals", "range_1m_pct"]),
    ("Events & scores", ["earnings_date", "days_to_earnings", "div_yield", "recom", "target_upside",
                         "score", "score_value", "score_quality", "score_momentum"]),
]

ETF_DETAIL_FIELDS = [
    ("Fund size & expenses", ["aum", "expense_ratio", "div_yield"]),
    ("Returns", ["perf_w", "perf_m", "perf_q", "perf_h", "perf_ytd", "perf_y"]),
    ("Technical", ["rsi", "rsi_w", "sma50_pct", "sma150_pct", "sma200_pct", "high52_pct", "range52", "atr_pct",
                   "beta", "rel_volume", "avg_volume"]),
]


def snapshot_table(row: pd.Series) -> str:
    """Compact Finviz-style key/value table (HTML, no scripts)."""
    end = "left" if L() == I.HE else "right"                 # values sit at the far end of the row
    out = ['<table class="hs-snapshot" style="width:100%;border-collapse:collapse;font-size:14px">']
    fields = ETF_DETAIL_FIELDS if row.get("sec_type") == "etf" else DETAIL_FIELDS
    for title, keys in fields:
        out.append(f'<tr><td colspan="2" style="padding:12px 4px 3px;font-weight:600;color:{theme.ACCENT};'
                   f'border-bottom:1px solid {theme.RULE}">{html.escape(t(title))}</td></tr>')
        for k in keys:
            v = fmt_value(k, row.get(k))
            color = ""
            if k in export._SIGNED and v.startswith(("+", "-")) and v not in ("+0.00%", "-0.00%"):
                color = f"color:{theme.GAIN};" if v.startswith("+") else f"color:{theme.LOSS};"
            out.append(f'<tr title="{html.escape(COLUMNS[k].help)}">'
                       f'<td style="padding:3px 4px;color:{theme.MUTED};border-bottom:1px solid {theme.HAIRLINE}">'
                       f'{html.escape(col_label(k))}</td>'
                       f'<td dir="ltr" style="padding:3px 4px;text-align:{end};font-weight:600;{color}'
                       f'border-bottom:1px solid {theme.HAIRLINE}">{html.escape(v)}</td></tr>')
    out.append("</table>")
    return "".join(out)


def range52_bar(price: float, low: float, high: float) -> str:
    """Where the price sits between the 52-week low and high; "" when that cannot be drawn honestly."""
    if pd.isna(price) or pd.isna(low) or pd.isna(high) or low >= high:
        return ""
    pct = max(0.0, min(100.0, (price - low) / (high - low) * 100))

    return (f'<div dir="ltr" style="display:flex;align-items:center;gap:6px;font-size:14px;color:{theme.MUTED};'
            f'margin-bottom:8px">'
            f'<span>{low:.2f}</span>'
            f'<div style="flex:1;height:4px;background:{theme.HAIRLINE};border-radius:2px;position:relative">'
            f'<div style="position:absolute;left:{pct}%;top:-3px;width:6px;height:10px;'
            f'background:{theme.ACCENT};border-radius:2px;transform:translateX(-50%)"></div>'
            f'</div><span>{high:.2f}</span></div>')


def detail(row: pd.Series) -> None:
    st.divider()
    chart, info = st.columns([3, 1.25])
    with chart:
        tradingview_chart(row["tv_symbol"])
    with info.container(key="detail"):
        st.markdown(f"### {row['ticker']}")
        sector = _clean_str(I.value("sector", row.get("sector"), L()))
        industry = _clean_str(row.get("industry"))
        country = _clean_str(I.value("country", row.get("country"), L()))
        company = escape_markdown(row.get("company"))
        lrm = "\u200e"     # keeps a trailing "." of English names ("Apple Inc.") in place in RTL
        st.caption(f"{company}{lrm}  \n{sector} · {industry}{lrm} · {country}"
                   + (f"  \n{row['indexes']}" if row.get("indexes") and not pd.isna(row.get("indexes")) else ""))
        p_val = row.get("price")
        p_delta = None if pd.isna(row.get("change")) else f"{row['change']:+.2f}%"
        st.metric(t("Price"), fmt_value("price", p_val), p_delta)
        bar = range52_bar(row.get("price"), row.get("low52"), row.get("high52"))
        if bar:
            st.caption(f"**{t('52-Week Range')}**")
            st.html(bar)
        if row.get("signals") and not pd.isna(row.get("signals")):
            st.caption(f"**{t('Signals')}:** {I.signal_tags(row['signals'], L())}")
        st.html(snapshot_table(row))
        l1, l2 = st.columns(2)
        l1.link_button("TradingView ↗", export.tradingview_url(row["tv_symbol"]), use_container_width=True)
        l2.link_button("Finviz ↗", f"https://finviz.com/quote.ashx?t={row['ticker']}", use_container_width=True)
        add_to_watchlist_button(row["ticker"])
    explain_panel.render(row, selections())
    sizing_panel.render(row)


def add_to_watchlist_button(ticker: str) -> None:
    watchlists = [n for n, p in PF.load_all().items() if p.kind == PF.WATCHLIST]
    options = watchlists + [NEW_WATCHLIST]
    if ss.get("add_wl") not in options:
        ss["add_wl"] = options[0]
    with st.popover(t("Add to watchlist"), icon=":material/playlist_add:", use_container_width=True):
        st.selectbox(t("Watchlist"), options, key="add_wl", format_func=lambda n: t(n) if n == NEW_WATCHLIST else n)
        if ss.get("add_wl") == NEW_WATCHLIST:
            st.text_input(t("List name"), key="add_wl_name")
        st.button(f"{t('Add')} {ticker}", on_click=on_add_to_watchlist, args=(ticker,), type="primary",
                  key="add_wl_btn")


def export_bar(res: pd.DataFrame, view_cols: list[str], snap) -> None:
    st.divider()
    title = ss.get("preset").removeprefix(USER_PREFIX) if ss.get("preset") not in (None, CUSTOM) else "custom"
    stem = export.file_stem(title)
    c0, c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.2, 1.2], vertical_alignment="center")
    all_cols = c0.toggle(t("Export all columns"), key="export_all",
                         help="כבוי: רק העמודות שבתצוגה הנוכחית. דלוק: כל השדות (כ-110 עמודות; "
                              "תבניות הנרות מסוכמות בעמודת Signals)")
    cols = [k for k, c in COLUMNS.items() if k != "tv_symbol" and c.kind != "flag"] if all_cols else view_cols
    filters_desc = described()
    if ss.get("query"):
        filters_desc = filters_desc + [(t("Search"), ss.get("query"))]
    headers = {k: col_label(k) for k in cols}
    preset = ss.get("preset")
    shown_title = (t("(custom)") if preset in (None, CUSTOM) else title if preset.startswith(USER_PREFIX)
                   else I.preset(title, L()))
    if preset_modified(all_presets()):
        shown_title += f" {t('(changed)')}"

    def xlsx() -> bytes:
        data = export.to_excel(display_values(res), cols, filters_desc, snap.fetched_at, shown_title,
                               headers=headers, t=t, rtl=L() == I.HE)
        export.save_copy(data, f"{stem}.xlsx")
        return data

    def csv() -> bytes:
        data = export.to_csv(display_values(res), cols, headers=headers)
        export.save_copy(data, f"{stem}.csv")
        return data

    def watchlist() -> str:
        data = export.to_watchlist(res)
        export.save_copy(data, f"{stem}_tradingview.txt")
        return data

    empty = res.empty
    c1.download_button("Excel", xlsx, f"{stem}.xlsx", on_click="ignore", disabled=empty,
                       icon=":material/table_view:",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True, help="קובץ אקסל מעוצב + גיליון עם הפילטרים שהופעלו")
    c2.download_button("CSV", csv, f"{stem}.csv", mime="text/csv", on_click="ignore", disabled=empty,
                       icon=":material/description:", use_container_width=True)
    c3.download_button(t("TradingView list"), watchlist, f"{stem}_tradingview.txt", mime="text/plain",
                       on_click="ignore", disabled=empty, use_container_width=True, icon=":material/list_alt:",
                       help="ייבוא ל-TradingView: Watchlist ← Import list. המניות מחולקות לפי סקטורים")
    with c4.popover(t("Tickers"), icon=":material/content_copy:", use_container_width=True, disabled=empty):
        st.caption("לחץ על אייקון ההעתקה בפינה")
        st.code(export.tickers_text(res), language=None, wrap_lines=True)
    hebrew_note(f"{t('A copy of every downloaded file is also saved in:')} {export.EXPORT_DIR}")
