"""
HalgOM Screener — Streamlit UI.

Run:  run_screener.bat   (or: streamlit run screener/app.py from the HalgOM folder)
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import logging.handlers
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from screener import export  # noqa: E402
from screener import groups as G  # noqa: E402
from screener import history  # noqa: E402
from screener import presets as P  # noqa: E402
from screener.data import Store, load_universe  # noqa: E402
from screener.data.store import DATA_DIR  # noqa: E402
from screener.data.tradingview import fetch_market, fetch_universe  # noqa: E402
from screener.derive import add_derived  # noqa: E402
from screener.fields import COLUMNS, DEFAULT_SORT, GROUP_COLUMNS, VIEWS  # noqa: E402
from screener.filters import (ANY, FILTERS, GROUPS, JUNK_RULES, active, describe, hide_junk,  # noqa: E402
                              option_labels, screen, search)
from screener.market import compute_regime, trend_strength  # noqa: E402

VERSION = "0.2.0"
MAX_AGE = dt.timedelta(minutes=15)       # TradingView data is ~15 min delayed anyway
RETRY_AFTER = dt.timedelta(minutes=2)    # when offline, don't hammer the network on every click
CUSTOM = "— Custom —"
USER_PREFIX = "★ "
GRID = 6
PAGES = ["Screener", "Groups", "Heatmap"]
# provider columns a cached snapshot must have; an older cache without them is refetched
REQUIRED = [k for k, c in COLUMNS.items() if c.tv]
# widget keys that must survive while their widget is not on the current page
PERSIST = ["preset", "query", "sort_col", "sort_asc", "view", "hide_junk", "only_new", "export_all",
           "g_by", "g_weight", "g_sector", "g_results", "h_universe", "h_color"]

st.set_page_config(page_title="HalgOM Screener", page_icon="📈", layout="wide")
ss = st.session_state
log = logging.getLogger("screener.app")


# ─────────────────────────── setup ─────────────────────────────────────────

@st.cache_resource
def _setup_logging() -> Path:
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(log_dir / "screener.log", maxBytes=1_000_000,
                                                   backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    root = logging.getLogger("screener")
    root.setLevel(logging.INFO)
    for h in [h for h in root.handlers if getattr(h, "_screener", False)]:  # survives Streamlit reloads
        root.removeHandler(h)
        h.close()
    handler._screener = True
    root.addHandler(handler)
    root.info("screener %s started", VERSION)
    return log_dir / "screener.log"


@st.cache_resource
def _store() -> Store:
    return Store(DATA_DIR)


@st.cache_data(ttl=MAX_AGE, show_spinner=False)
def _market():
    try:
        m = fetch_market()
        return m, compute_regime(m)
    except Exception:
        log.exception("market fetch failed")
        return None, None


def get_data(force: bool = False):
    now = dt.datetime.now()
    snap = ss.get("snap")
    due = (force or snap is None
           or (snap.source == "stale" and now - ss.get("last_attempt", now) > RETRY_AFTER)
           or (snap.source != "stale" and snap.age > MAX_AGE))
    if due:
        ss.last_attempt = now
        with st.spinner("Loading market data from TradingView… (≈15 sec)"):
            try:
                snap = load_universe(_store(), fetch_universe, MAX_AGE, force=force, required=REQUIRED)
            except Exception as e:
                log.exception("no data available")
                st.error("לא ניתן להוריד נתונים, ואין נתונים שמורים מהפעם הקודמת. "
                         "בדוק את החיבור לאינטרנט ונסה שוב.")
                st.caption(f"Details: {e}")
                if st.button("Try again", type="primary"):
                    st.rerun()
                st.stop()
        ss.snap = snap
        ss.df = add_derived(snap.df)
    return ss.snap, ss.df


def get_previous(snap) -> history.Previous | None:
    """Previous day's snapshot (derived), loaded once per snapshot day."""
    day = snap.fetched_at.date()
    if ss.get("prev_for") != day:
        ss.prev = history.load_previous(_store(), day)
        ss.prev_for = day
    return ss.prev


def all_presets() -> dict[str, P.Preset]:
    out = dict(P.BUILTIN)
    out.update({USER_PREFIX + n: p for n, p in P.load_user().items()})
    return out


def selections() -> dict[str, str]:
    return {f.key: ss.get(f"f_{f.key}", ANY) for f in FILTERS}


def keep_widget_state() -> None:
    """Streamlit drops a widget's value when the widget is not rendered (e.g. on another page)."""
    for k in PERSIST + [f"f_{f.key}" for f in FILTERS]:
        if k in ss:
            ss[k] = ss[k]


# ─────────────────────────── callbacks ─────────────────────────────────────

def on_preset():
    p = all_presets().get(ss.preset)
    if p is None:
        return
    for f in FILTERS:
        ss[f"f_{f.key}"] = p.filters.get(f.key, ANY)
    ss.view = p.view if p.view in VIEWS else "Overview"
    ss.sort_col = p.sort if p.sort in COLUMNS else DEFAULT_SORT[0]
    ss.sort_asc = p.ascending
    ss.query = ""


def on_reset():
    for f in FILTERS:
        ss[f"f_{f.key}"] = ANY
    ss.preset = CUSTOM
    ss.query = ""


def on_save():
    name = (ss.get("save_name") or "").strip()
    if not name:
        return
    P.save_user(name, P.Preset(filters=active(selections()), view=ss.view,
                               sort=ss.sort_col, ascending=ss.sort_asc))
    ss.preset = USER_PREFIX + name
    ss.save_name = ""
    st.toast(f"Saved screen “{name}”", icon="💾")


def on_delete():
    name = ss.get("delete_name")
    if name:
        P.delete_user(name)
        if ss.get("preset") == USER_PREFIX + name:
            ss.preset = CUSTOM
        st.toast(f"Deleted “{name}”", icon="🗑️")


# ─────────────────────────── rendering helpers ─────────────────────────────

def column_config(key: str):
    c = COLUMNS[key]
    cc = st.column_config
    kw = {"label": c.label, "help": c.help}
    match c.kind:
        case "text":
            width = {"signals": 300, "indexes": 240, "new": 60}.get(key)
            return cc.TextColumn(pinned=key == "ticker", width=width, **kw)
        case "price":
            return cc.NumberColumn(format="%.2f", **kw)
        case "pct":
            return cc.NumberColumn(format="%.2f%%", **kw)
        case "ratio":
            return cc.NumberColumn(format="%.2f", **kw)
        case "int":
            return cc.NumberColumn(format="localized", **kw)
        case "money" | "count":
            return cc.NumberColumn(format="compact", **kw)
        case "date":
            return cc.DateColumn(format="YYYY-MM-DD", **kw)
        case "days":
            return cc.NumberColumn(format="%d", **kw)
        case "score":
            return cc.ProgressColumn(min_value=0, max_value=100, format="%d", **kw)
        case "range":
            return cc.ProgressColumn(min_value=0, max_value=1, format="percent", **kw)
        case "flag":
            return cc.NumberColumn(format="%d", **kw)
    return cc.Column(**kw)


def hebrew_note(text: str) -> None:
    st.html(f'<div dir="rtl" style="text-align:right;font-size:0.85rem;color:#6b7280">{html.escape(text)}</div>')


def _sign_color(v) -> str:
    if isinstance(v, (int, float)) and not pd.isna(v) and v != 0:
        return "color: #1a7f37" if v > 0 else "color: #cf222e"
    return ""


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
        return "yes" if v else "no"
    return str(v)


def tradingview_chart(symbol: str, height: int = 560) -> None:
    config = {
        "autosize": False, "width": "100%", "height": height - 34, "symbol": symbol, "interval": "D", "timezone": "America/New_York",
        "theme": "light", "style": "1", "locale": "en", "allow_symbol_change": True,
        "hide_side_toolbar": False, "details": False, "calendar": False,
        "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies"],
        "support_host": "https://www.tradingview.com",
    }
    components.html(f"""
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


def header(snap, n_universe: int) -> None:
    left, mid, right = st.columns([2.2, 5, 2.3], vertical_alignment="center")
    with left:
        st.markdown("### 📈 HalgOM Screener")
        st.caption(f"v{VERSION} · US stocks · {n_universe:,} in universe")
    with mid:
        market, regime = _market()
        if market is None:
            st.caption("Market data unavailable")
        else:
            cols = st.columns([1.3, 1, 1, 1, 1])
            colors = {"bull": "green", "bear": "red", "sideways": "orange", "unknown": "gray"}
            with cols[0]:
                st.badge(regime.label.upper(), color=colors[regime.label], icon=":material/monitoring:",
                         help="מצב השוק לפי כללי HalgOM:\n\n" + "\n".join(f"- {r}" for r in regime.reasons))
                spy = market.loc["SPY"]
                trend = trend_strength(spy.get("adx", float("nan")), spy.get("di_plus", float("nan")),
                                       spy.get("di_minus", float("nan")))
                st.badge(trend.label, color=trend.color, icon=":material/trending_up:", help=trend.help)
                st.caption("Market regime · SPY ADX")
            for c, name in zip(cols[1:], ["SPY", "QQQ", "VIX", "HYG"]):
                row = market.loc[name]
                vs200 = "" if pd.isna(row.sma200) or name == "VIX" else (
                    " · ▲SMA200" if row.close > row.sma200 else " · ▼SMA200")
                c.metric(name, f"{row.close:,.2f}", f"{row.change:+.2f}%",
                         delta_color="inverse" if name == "VIX" else "normal",
                         help=f"{name} vs SMA20/50/200: {row.sma20:,.2f} / {row.sma50:,.2f} / {row.sma200:,.2f}")
                if vs200:
                    c.caption(vs200.strip(" ·"))
    with right:
        src = {"live": "live", "cache": "cached", "stale": "OFFLINE"}[snap.source]
        st.caption(f"Data as of **{snap.fetched_at:%Y-%m-%d %H:%M}** ({src}) · {source_text(snap.meta)}",
                   help="מקור הנתונים ומצב העדכון כפי שהספק מדווח. delayed = בעיכוב, realtime = בזמן אמת")
        if st.button("🔄 Refresh data", use_container_width=True):
            get_data(force=True)
            _market.clear()
            st.rerun()
    if snap.source == "stale":
        st.warning(f"⚠️ לא ניתן להתחבר ל-TradingView כרגע — מוצגים נתונים שמורים מ-"
                   f"{snap.fetched_at:%d/%m/%Y} בשעה {snap.fetched_at:%H:%M}. ננסה שוב אוטומטית בעוד כמה דקות.")


def source_text(meta: dict | None) -> str:
    """'TradingView · delayed 15 min' from snapshot metadata (older caches have none)."""
    meta = meta or {}
    provider = {"tradingview": "TradingView"}.get(meta.get("provider"), meta.get("provider") or "TradingView")
    mode = meta.get("update_mode")
    if mode == "realtime":
        return f"{provider} · realtime"
    if mode == "delayed" and meta.get("delay_min"):
        return f"{provider} · delayed {meta['delay_min']} min"
    return f"{provider} · delayed ~15 min"


def junk_toggle(container) -> None:
    container.toggle("Hide junk", key="hide_junk",
                     help="מסתיר מניות זבל: " + ", ".join(label for label, _ in JUNK_RULES)
                          + ". מניה בלי נתון לא מוסתרת")


def nav() -> str:
    st.segmented_control("Page", PAGES, key="page", label_visibility="collapsed", required=True,
                         help="Screener = סריקה, Groups = ביצועים לפי סקטור ותעשייה, Heatmap = מפת השוק")
    return ss.page


def toolbar(presets: dict[str, P.Preset]) -> None:
    names = [CUSTOM] + list(presets)
    if ss.get("preset") not in names:
        ss.preset = CUSTOM
    c1, c2, c3, c4, c7, c5, c6 = st.columns([2.6, 2.4, 1.8, 1.1, 1.2, 1, 1], vertical_alignment="bottom")
    c1.selectbox("Screen", names, key="preset", on_change=on_preset,
                 help="סריקות מוכנות. בחירה בסריקה ממלאת את הפילטרים — אפשר לשנות אותם אחר כך")
    c2.text_input("Search", key="query", placeholder="Ticker / company, or list: AAPL, MSFT",
                  help="חיפוש לפי סימול או שם חברה. אפשר להדביק רשימת סימולים מופרדת בפסיקים")
    sortable = [k for k, c in COLUMNS.items() if c.kind not in ("text", "flag") or k in ("ticker", "company")]
    c3.selectbox("Order by", sortable, key="sort_col", format_func=lambda k: COLUMNS[k].label)
    c4.toggle("Ascending", key="sort_asc")
    junk_toggle(c7)
    c5.button("Reset", on_click=on_reset, use_container_width=True, help="ניקוי כל הפילטרים")
    with c6.popover("💾", use_container_width=True, help="שמירת הסריקה הנוכחית / מחיקת סריקה שמורה"):
        st.text_input("Save current screen as", key="save_name")
        st.button("Save", on_click=on_save, type="primary")
        user = P.load_user()
        if user:
            st.divider()
            st.selectbox("Delete saved screen", list(user), key="delete_name")
            st.button("Delete", on_click=on_delete)
    note = presets.get(ss.preset).note if ss.preset in presets else ""
    if note:
        hebrew_note(f"ℹ️ {note}")


def filter_panel(df: pd.DataFrame) -> None:
    sel = selections()
    n_active = len(active(sel))
    with st.expander(f"Filters ({n_active} active)" if n_active else "Filters", expanded=True):
        tabs = st.tabs([f"{g} ({sum(1 for f in FILTERS if f.group == g and sel[f.key] != ANY)})"
                        if any(f.group == g and sel[f.key] != ANY for f in FILTERS) else g for g in GROUPS])
        for tab, group in zip(tabs, GROUPS):
            with tab:
                group_filters = [f for f in FILTERS if f.group == group]
                for i in range(0, len(group_filters), GRID):
                    cols = st.columns(GRID)
                    for col, f in zip(cols, group_filters[i:i + GRID]):
                        labels = option_labels(f, df)
                        key = f"f_{f.key}"
                        if ss.get(key, ANY) not in labels:
                            ss[key] = ANY
                        col.selectbox(f.label, labels, key=key, help=f.tooltip)


def results(df: pd.DataFrame, res: pd.DataFrame, snap, prev: history.Previous | None) -> None:
    universe = hide_junk(df, ss.hide_junk)
    top = st.columns([3, 5, 1.6], vertical_alignment="center")
    top[0].markdown(f"#### {len(res):,} results" + (f" / {len(universe):,}" if len(res) != len(universe) else ""))
    with top[1]:
        st.segmented_control("View", list(VIEWS), key="view", label_visibility="collapsed", required=True)
    view_cols = VIEWS[ss.view]
    notes = [f"**{k}:** {v}" for k, v in describe(selections())]
    if ss.hide_junk and len(universe) < len(df):
        notes.append(f"{len(df) - len(universe):,} junk stocks hidden")
    if prev is not None:
        n_new = int((res["new"] == history.NEW).sum())
        top[2].toggle(f"Only new ({n_new:,})", key="only_new",
                      help=f"רק מניות שנכנסו לתוצאות מאז ה-snapshot של {prev.day:%d/%m/%Y}")
        view_cols = [view_cols[0], "new", *view_cols[1:]]
        notes.append(f"NEW = not in these results on {prev.day:%Y-%m-%d}")
        if ss.get("only_new"):
            res = res[res["new"] == history.NEW]
    if notes:
        st.caption(" · ".join(notes))
    if prev is not None and prev.missing:
        hebrew_note(f"ההשוואה ל-{prev.day:%d/%m/%Y} חלקית: ה-snapshot נשמר בגרסה קודמת בלי {len(prev.missing)} "
                    "שדות חדשים, ולכן פילטרים על השדות האלה מסמנים יותר מניות כ-NEW. יסתדר מעצמו ממחר.")

    table = res[view_cols]
    signed = [c for c in view_cols if c in export._SIGNED]
    if signed:
        table = table.style.map(_sign_color, subset=signed)
    event = st.dataframe(
        table, hide_index=True, height=560, key="table",
        column_config={k: column_config(k) for k in view_cols},
        on_select="rerun", selection_mode="single-row", placeholder="—",
    )
    rows = event.selection.rows if event and event.selection else []
    if rows and rows[0] < len(res):
        detail(res.iloc[rows[0]])
    else:
        hebrew_note("👆 לחץ על שורה בטבלה כדי לפתוח גרף ונתונים של המניה")
    export_bar(res, view_cols, snap)


DETAIL_FIELDS = [
    ("Valuation", ["market_cap", "pe", "fwd_pe", "peg", "ps", "pb", "ev_ebitda"]),
    ("Growth & quality", ["eps_growth", "eps_growth_q", "sales_growth_q", "roe", "roic", "net_margin", "debt_eq"]),
    ("Technical", ["rsi", "rsi_w", "sma50_pct", "sma150_pct", "sma200_pct", "high52_pct", "range52", "atr_pct",
                   "beta", "rel_volume"]),
    ("Signals", ["n_signals", "range_1m_pct"]),
    ("Events & scores", ["earnings_date", "days_to_earnings", "div_yield", "recom", "target_upside",
                         "score", "score_value", "score_quality", "score_momentum"]),
]


def snapshot_table(row: pd.Series) -> str:
    """Compact Finviz-style key/value table (HTML, no scripts)."""
    out = ['<table style="width:100%;border-collapse:collapse;font-size:13px">']
    for title, keys in DETAIL_FIELDS:
        out.append(f'<tr><td colspan="2" style="padding:10px 4px 3px;font-weight:600;'
                   f'border-bottom:1px solid #d0d7de">{html.escape(title)}</td></tr>')
        for k in keys:
            v = fmt_value(k, row.get(k))
            color = ""
            if k in export._SIGNED and v.startswith(("+", "-")) and v not in ("+0.00%", "-0.00%"):
                color = "color:#1a7f37;" if v.startswith("+") else "color:#cf222e;"
            out.append(f'<tr title="{html.escape(COLUMNS[k].help)}">'
                       f'<td style="padding:2px 4px;color:#57606a;border-bottom:1px solid #eef1f4">'
                       f'{html.escape(COLUMNS[k].label)}</td>'
                       f'<td style="padding:2px 4px;text-align:right;font-weight:600;{color}'
                       f'border-bottom:1px solid #eef1f4">{html.escape(v)}</td></tr>')
    out.append("</table>")
    return "".join(out)


def detail(row: pd.Series) -> None:
    st.divider()
    chart, info = st.columns([3, 1.25])
    with chart:
        tradingview_chart(row["tv_symbol"])
    with info:
        st.markdown(f"### {row['ticker']}")
        st.caption(f"{row['company']}  \n{row['sector']} · {row['industry']} · {row['country']}"
                   + (f"  \n{row['indexes']}" if row.get("indexes") else ""))
        st.metric("Price", fmt_value("price", row["price"]), f"{row['change']:+.2f}%")
        if row.get("signals"):
            st.caption(f"**Signals:** {row['signals']}")
        st.html(snapshot_table(row))
        st.markdown(f"[Open in TradingView ↗]({export.tradingview_url(row['tv_symbol'])}) · "
                    f"[Finviz ↗](https://finviz.com/quote.ashx?t={row['ticker']})")


def export_bar(res: pd.DataFrame, view_cols: list[str], snap) -> None:
    st.divider()
    title = ss.preset.removeprefix(USER_PREFIX) if ss.get("preset") not in (None, CUSTOM) else "custom"
    stem = export.file_stem(title)
    c0, c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.2, 1.2], vertical_alignment="center")
    all_cols = c0.toggle("Export all columns", key="export_all",
                         help="כבוי: רק העמודות שבתצוגה הנוכחית. דלוק: כל השדות (כ-110 עמודות; "
                              "תבניות הנרות מסוכמות בעמודת Signals)")
    cols = [k for k, c in COLUMNS.items() if k != "tv_symbol" and c.kind != "flag"] if all_cols else view_cols
    filters_desc = describe(selections())
    if ss.get("query"):
        filters_desc = filters_desc + [("Search", ss.query)]

    def xlsx() -> bytes:
        data = export.to_excel(res, cols, filters_desc, snap.fetched_at, title)
        export.save_copy(data, f"{stem}.xlsx")
        return data

    def csv() -> bytes:
        data = export.to_csv(res, cols)
        export.save_copy(data, f"{stem}.csv")
        return data

    def watchlist() -> str:
        data = export.to_watchlist(res)
        export.save_copy(data, f"{stem}_tradingview.txt")
        return data

    empty = res.empty
    c1.download_button("📊 Excel", xlsx, f"{stem}.xlsx", on_click="ignore", disabled=empty,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True, help="קובץ אקסל מעוצב + גיליון עם הפילטרים שהופעלו")
    c2.download_button("📄 CSV", csv, f"{stem}.csv", mime="text/csv", on_click="ignore", disabled=empty,
                       use_container_width=True)
    c3.download_button("📋 TradingView list", watchlist, f"{stem}_tradingview.txt", mime="text/plain",
                       on_click="ignore", disabled=empty, use_container_width=True,
                       help="ייבוא ל-TradingView: Watchlist ← Import list. המניות מחולקות לפי סקטורים")
    with c4.popover("🔤 Tickers", use_container_width=True, disabled=empty):
        st.caption("לחץ על אייקון ההעתקה בפינה")
        st.code(export.tickers_text(res), language=None, wrap_lines=True)
    hebrew_note(f"עותק של כל קובץ שמורד נשמר גם בתיקייה: {export.EXPORT_DIR}")


# ─────────────────────────── Groups / Heatmap pages ────────────────────────

HEAT_COLORS = {"change": 3, "perf_w": 6, "perf_m": 10, "perf_q": 20, "perf_h": 30, "perf_ytd": 40, "perf_y": 50}
HEAT_UNIVERSES = ["S&P 500", "Nasdaq 100", "Screener results", "All stocks (top 1,000)"]


def groups_page(df: pd.DataFrame, res: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns([1.6, 1.8, 2.4, 1.6, 1.2], vertical_alignment="bottom")
    ss.setdefault("g_by", "Sector")
    ss.setdefault("g_weight", "Equal")
    c1.segmented_control("Group by", ["Sector", "Industry"], key="g_by", required=True)
    c2.segmented_control("Averages", ["Equal", "Market cap"], key="g_weight", required=True,
                         help="Equal = כל מניה נספרת פעם אחת. Market cap = מניות גדולות משפיעות יותר (כמו מדד)")
    base = res if ss.get("g_results") else hide_junk(df, ss.hide_junk)
    by = ss.g_by.lower()
    if by == "industry":
        sectors = ["All sectors"] + sorted(base["sector"].dropna().astype(str).unique())
        if ss.get("g_sector") not in sectors:
            ss.g_sector = "All sectors"
        c3.selectbox("Sector", sectors, key="g_sector", help="הצגת התעשיות של סקטור אחד בלבד")
        if ss.g_sector != "All sectors":
            base = base[base["sector"] == ss.g_sector]
    c4.toggle("Only screener results", key="g_results",
              help="דלוק: הקבוצות מחושבות רק מהמניות שעברו את הפילטרים בדף Screener")
    junk_toggle(c5)
    table = G.group_performance(base, by=by, weight="cap" if ss.g_weight == "Market cap" else "equal")
    st.markdown(f"#### {len(table):,} {'sectors' if by == 'sector' else 'industries'} · {len(base):,} stocks")
    cc = st.column_config
    config = {"group": cc.TextColumn(GROUP_COLUMNS["group"].label, help=GROUP_COLUMNS["group"].help, pinned=True),
              "stocks": cc.NumberColumn(GROUP_COLUMNS["stocks"].label, help=GROUP_COLUMNS["stocks"].help, format="%d"),
              "market_cap": cc.NumberColumn(GROUP_COLUMNS["market_cap"].label, help=GROUP_COLUMNS["market_cap"].help,
                                            format="compact"),
              "rel_volume": cc.NumberColumn(GROUP_COLUMNS["rel_volume"].label, help=GROUP_COLUMNS["rel_volume"].help,
                                            format="%.2f")}
    for k in G.PERF + ["pct_above_sma50"]:
        config[k] = cc.NumberColumn(GROUP_COLUMNS[k].label, help=GROUP_COLUMNS[k].help, format="%.2f%%")
    styled = table.style.map(_sign_color, subset=G.PERF)
    st.dataframe(styled, hide_index=True, height=min(38 + 35 * len(table), 760), column_config=config,
                 key="groups_table")
    hebrew_note("ביצועים ממוצעים של המניות בכל קבוצה. לחיצה על כותרת עמודה ממיינת. "
                "מומלץ להשאיר את Hide junk דלוק — מניות זבל עם תנועות של מאות אחוזים מעוותות ממוצע שווה-משקל.")


def heatmap_page(df: pd.DataFrame, res: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([2.4, 2.4, 1.2], vertical_alignment="bottom")
    ss.setdefault("h_universe", HEAT_UNIVERSES[0])
    ss.setdefault("h_color", "change")
    c1.selectbox("Stocks", HEAT_UNIVERSES, key="h_universe",
                 help="אילו מניות להציג. S&P 500 מקביל למפת השוק של Finviz")
    c2.selectbox("Color by", list(HEAT_COLORS), key="h_color", format_func=lambda k: COLUMNS[k].label,
                 help="ירוק = עלייה, אדום = ירידה. הצבע רווי בקצוות הסקאלה (למשל ±3% לשינוי יומי)")
    junk_toggle(c3)
    universe = hide_junk(df, ss.hide_junk)
    match ss.h_universe:
        case "Screener results":
            data = res
        case "All stocks (top 1,000)":
            data = universe.nlargest(1000, "market_cap")
        case label:
            data = universe[universe["indexes"].fillna("").astype(str).str.contains(label, regex=False)]
    if data.empty:
        st.info("אין מניות להצגה")
        return
    color = ss.h_color
    st.plotly_chart(G.heatmap_figure(data, color, limit=HEAT_COLORS[color]), key="heatmap",
                    config={"displaylogo": False})
    hebrew_note(f"{len(data):,} מניות. גודל = שווי שוק, צבע = {COLUMNS[color].label}. "
                "לחיצה על סקטור או תעשייה מגדילה אותם; לחזרה לחץ על שורת הניווט למעלה.")


# ─────────────────────────── page ──────────────────────────────────────────

CSS = """
<style>
  .block-container {padding-top: 3.5rem; padding-bottom: 2rem; max-width: 1800px;}  /* clear the fixed top bar */
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""


def main() -> None:
    _setup_logging()
    st.html(CSS)
    keep_widget_state()
    ss.setdefault("page", PAGES[0])
    ss.setdefault("view", "Overview")
    ss.setdefault("sort_col", DEFAULT_SORT[0])
    ss.setdefault("sort_asc", DEFAULT_SORT[1])
    ss.setdefault("hide_junk", True)
    for f in FILTERS:
        ss.setdefault(f"f_{f.key}", ANY)

    snap, df = get_data()
    header(snap, len(df))
    page = nav()
    if page == "Screener":
        toolbar(all_presets())
        filter_panel(df)

    sel = selections()
    res = search(screen(df, sel, ss.hide_junk), ss.get("query", ""))
    prev = get_previous(snap)
    res = res.assign(new=history.mark_new(res, prev, sel, ss.hide_junk))
    res = res.sort_values(ss.sort_col, ascending=ss.sort_asc, na_position="last")
    if page == "Groups":
        groups_page(df, res)
    elif page == "Heatmap":
        heatmap_page(df, res)
    else:
        results(df, res, snap, prev)
    hebrew_note("נתונים: TradingView (בעיכוב של כ-15 דקות). הכלי לצורכי מחקר בלבד ואינו ייעוץ השקעות.")


main()
