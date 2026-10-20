"""
HalgOM Screener — Streamlit UI.

Run:  run_screener.bat   (or: streamlit run screener/app.py from the HalgOM folder)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import logging
import logging.handlers
import re
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
from screener import i18n as I  # noqa: E402
from screener import lifecycle  # noqa: E402
from screener import portfolio as PF  # noqa: E402
from screener import presets as P  # noqa: E402
from screener import theme  # noqa: E402
from screener.data import FETCH_LOCK, Store, load_universe  # noqa: E402
from screener.data.store import DATA_DIR  # noqa: E402
from screener.data.tradingview import fetch_market, fetch_universe  # noqa: E402
from screener.derive import add_derived  # noqa: E402
from screener.fields import COLUMNS, DEFAULT_SORT, GROUP_COLUMNS, VIEWS, provider_columns  # noqa: E402
from screener.filters import (ANY, FILTERS, FILTERS_BY_KEY, GROUPS, JUNK_RULES, active,  # noqa: E402
                              apply_filters, hide_junk, in_index, option_labels, search)
from screener.market import compute_regime, trend_strength  # noqa: E402

VERSION = "0.2.0"
MAX_AGE = dt.timedelta(minutes=15)       # TradingView data is ~15 min delayed anyway
RETRY_AFTER = dt.timedelta(minutes=2)    # when offline, don't hammer the network on every click
RETRY_TIMEOUT = 20                         # seconds, for those retries (a normal fetch takes ~15 s)
REFRESH_MIN = dt.timedelta(seconds=60)   # "Refresh data" clicked again sooner: no new request
CUSTOM = "— Custom —"
USER_PREFIX = "★ "
GRID = 6
PAGES = ["Screener", "Groups", "Heatmap", "Portfolios", "Guide"]
GUIDE = Path(__file__).with_name("guide.html")
# provider columns a cached snapshot must have; an older cache without them is refetched
REQUIRED = provider_columns()
# widget keys that must survive while their widget is not on the current page
PERSIST = ["lang", "preset", "query", "sort_col", "sort_asc", "view", "hide_junk", "only_new", "export_all",
           "g_by", "g_weight", "g_sector", "g_results", "h_universe", "h_color", "large_text", "pf_name"]

st.set_page_config(page_title="HalgOM Screener", page_icon="📈", layout="wide")
ss = st.session_state
log = logging.getLogger("screener.app")


def t(text: str) -> str:
    """Display text in the chosen language (i18n.py). Internal values stay English."""
    return I.ui(text, ss.get("lang", I.HE))


def L() -> str:
    return ss.get("lang", I.HE)


def opt(text: str) -> str:
    return I.option(text, L())


def col_label(key: str) -> str:
    return I.column(key, COLUMNS[key].label, L())


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
def _watchdog():
    """One per server: stops it after 15 minutes with no open browser tab (lifecycle.py)."""
    return lifecycle.start_idle_watchdog()


@st.cache_resource
def _store() -> Store:
    return Store(DATA_DIR)


@st.cache_data(ttl=MAX_AGE, show_spinner=False)
def _fetch_market():
    with FETCH_LOCK:
        return fetch_market()


def _market():
    try:
        m = _fetch_market()
        return m, compute_regime(m)
    except Exception:
        log.exception("market fetch failed")
        return None, None


_market.clear = _fetch_market.clear


def get_data(force: bool = False):
    now = dt.datetime.now()
    snap = ss.get("snap")
    due = (force or snap is None
           or (snap.source == "stale" and now - ss.get("last_attempt", now) > RETRY_AFTER)
           or (snap.source != "stale" and not dt.timedelta(0) <= snap.age <= MAX_AGE))   # < 0: clock went back
    if due:
        ss.last_attempt = now
        with st.spinner(t("Loading market data from TradingView… (≈15 sec)")):
            try:
                retry = snap is not None and snap.source == "stale"     # offline: don't block a click for 60 s
                snap = load_universe(_store(), fetch_universe, MAX_AGE, force=force, required=REQUIRED,
                                     timeout=RETRY_TIMEOUT if retry else None)
            except Exception as e:
                log.exception("no data available")
                st.error("לא ניתן להוריד נתונים, ואין נתונים שמורים מהפעם הקודמת. "
                         "בדוק את החיבור לאינטרנט ונסה שוב.")
                st.caption(f"{t('Details')}: {e}")
                if st.button(t("Try again"), type="primary"):
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
        ss.prev_hits = {}
    return ss.prev


def new_column(res: pd.DataFrame, prev: history.Previous | None, sel: dict[str, str]) -> pd.Series:
    """'NEW' tags; yesterday's matches are computed once per (filters, junk switch), not on every click."""
    if prev is None:
        return history.mark_new(res, None)
    key = (tuple(sorted(active(sel).items())), ss.hide_junk)
    if key not in ss.prev_hits:
        ss.prev_hits[key] = history.previous_tickers(prev, sel, ss.hide_junk)
    return history.mark_new(res, ss.prev_hits[key])


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


def all_presets() -> dict[str, P.Preset]:
    out = dict(P.BUILTIN)
    out.update({USER_PREFIX + n: p for n, p in P.load_user().items()})
    return out


def selections() -> dict[str, str]:
    return {f.key: ss.get(f"f_{f.key}", ANY) for f in FILTERS}


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
        ss[f"f_{filter_of[removed]}"] = ANY


def display_values(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with data values in the display language (sector, country, the NEW tag) for tables and exports."""
    if L() != I.HE:
        return df
    out = df.copy()
    for c in ("sector", "country", "group"):
        if c in out.columns:
            out[c] = out[c].map(lambda v, c=c: I.value("sector" if c == "group" else c, v, I.HE))
    if "new" in out.columns:
        out["new"] = out["new"].replace(history.NEW, t("NEW"))
    if "signals" in out.columns:
        out["signals"] = out["signals"].map(lambda v: I.signal_tags(v, L()) if isinstance(v, str) else v)
    return out


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


def on_clear_all():
    """The empty-results button: every filter, the search and "Only new"."""
    on_reset()
    ss.only_new = False


def on_save():
    name = (ss.get("save_name") or "").strip()
    if not name:
        return
    try:
        P.save_user(name, P.Preset(filters=active(selections()), view=ss.view,
                                   sort=ss.sort_col, ascending=ss.sort_asc))
    except Exception as e:
        log.warning("cannot save screen %r: %s", name, e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    ss.preset = USER_PREFIX + name
    ss.save_name = ""
    st.toast(f"{t('Saved screen')}: “{name}”", icon="💾")


def on_exit():
    ss.exiting = True


def on_delete():
    name = ss.get("delete_name")
    if name:
        try:
            P.delete_user(name)
        except Exception as e:
            log.warning("cannot delete screen %r: %s", name, e)
            st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
            return
        if ss.get("preset") == USER_PREFIX + name:
            ss.preset = CUSTOM
        st.toast(f"{t('Deleted')}: “{name}”", icon="🗑️")


# ─────────────────────────── rendering helpers ─────────────────────────────

def column_config(key: str):
    c = COLUMNS[key]
    cc = st.column_config
    kw = {"label": col_label(key), "help": c.help}
    match c.kind:
        case "text":
            width_map = {
                "signals": 300, "indexes": 240, "new": 60,
                "ticker": "small", "exchange": "small", "country": "small", "sec_type": "small",
                "company": "medium", "sector": "medium", "industry": "medium"
            }
            width = width_map.get(key)
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
    st.html(f'<div dir="rtl" class="hs-note" style="text-align:right">{html.escape(text)}</div>')


def _sign_color(v) -> str:
    if isinstance(v, (int, float)) and not pd.isna(v) and v != 0:
        return f"color: {theme.GAIN}" if v > 0 else f"color: {theme.LOSS}"
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
        return ("כן" if v else "לא") if L() == I.HE else ("yes" if v else "no")
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


def trend_text(label: str) -> str:
    """'Trend strong ▲' -> 'מגמה חזקה ▲' (the arrow is kept)."""
    arrow = "▲" if "▲" in label else "▼" if "▼" in label else ""
    return f"{t(label.replace(arrow, '').strip() if arrow else label)} {arrow}".strip()


def header(snap, n_universe: int) -> None:
    with st.container(key="masthead"):
        _header(snap, n_universe)


def _header(snap, n_universe: int) -> None:
    left, mid, right = st.columns([2.2, 5, 2.3], vertical_alignment="center")
    with left:
        st.html('<div class="hs-brand"><i></i><b>HalgOM</b><span>Screener</span></div>')
        st.caption(f"v{VERSION} · {t('US stocks')} · {n_universe:,} {t('in universe')}")
        c1, c2 = st.columns([1, 1], vertical_alignment="center")
        c1.segmented_control("Language", list(I.LANGS), key="lang", format_func=I.LANGS.get, required=True,
                             label_visibility="collapsed")
        c2.toggle(t("Large text"), key="large_text")
    with mid:
        market, regime = _market()
        if market is None:
            st.caption(t("Market data unavailable"))
        else:
            cols = st.container(key="tape").columns([1.3, 1, 1, 1, 1])
            colors = {"bull": "green", "bear": "red", "sideways": "orange", "unknown": "gray"}
            with cols[0]:
                st.badge(t(regime.label.upper()), color=colors[regime.label], icon=":material/monitoring:",
                         help="מצב השוק לפי כללי HalgOM:\n\n" + "\n".join(f"- {r}" for r in regime.reasons))
                spy = market.loc["SPY"]
                trend = trend_strength(spy.get("adx", float("nan")), spy.get("di_plus", float("nan")),
                                       spy.get("di_minus", float("nan")))
                st.badge(trend_text(trend.label), color=trend.color, icon=":material/trending_up:", help=trend.help)
                st.caption(t("Market regime · SPY ADX"))
            for c, name in zip(cols[1:], ["SPY", "QQQ", "VIX", "HYG"]):
                row = market.loc[name]
                vs200 = "" if pd.isna(row.sma200) or name == "VIX" else (
                    " · \u200e▲SMA200" if row.close > row.sma200 else " · \u200e▼SMA200")   # LRM: arrow stays left in RTL
                c.metric(name, f"{row.close:,.2f}", f"{row.change:+.2f}%",
                         delta_color="inverse" if name == "VIX" else "normal",
                         help=f"{name} vs SMA20/50/200: {row.sma20:,.2f} / {row.sma50:,.2f} / {row.sma200:,.2f}")
                if vs200:
                    c.caption(vs200.removeprefix(" · "))
    with right:
        src = t({"live": "live", "cache": "cached", "stale": "OFFLINE"}[snap.source])
        st.caption(f"{t('Data time')}: **{snap.fetched_at:%H:%M}** · {snap.fetched_at:%d/%m/%Y} ({src}) · "
                   f"{source_text(snap.meta)}",
                   help="מקור הנתונים ומצב העדכון כפי שהספק מדווח. delayed = בעיכוב, realtime = בזמן אמת")
        r1, r2 = st.columns([1.7, 1.2])
        if r1.button(t("Refresh data"), icon=":material/refresh:", use_container_width=True):
            if snap.source != "stale" and snap.age < REFRESH_MIN:      # every click is a TradingView request
                st.toast(f"הנתונים עודכנו לפני {max(0, int(snap.age.total_seconds()))} שניות", icon="ℹ️")
            else:
                get_data(force=True)
                _market.clear()
                st.rerun()
        with r2.popover(t("Exit"), icon=":material/power_settings_new:", use_container_width=True,
                        help="סגירת הסקרינר"):
            hebrew_note("סוגר את הסקרינר ואת החלון השחור. בפעם הבאה פותחים שוב מהקיצור בשולחן העבודה.")
            st.button(t("Close the screener"), key="exit", type="primary", on_click=on_exit)
    if snap.source == "stale":
        st.warning(f"⚠️ לא ניתן להתחבר ל-TradingView כרגע — מוצגים נתונים שמורים מ-"
                   f"{snap.fetched_at:%d/%m/%Y} בשעה {snap.fetched_at:%H:%M}. ננסה שוב אוטומטית בעוד כמה דקות.")


def source_text(meta: dict | None) -> str:
    """'TradingView · delayed 15 min' from snapshot metadata (older caches have none)."""
    meta = meta or {}
    provider = {"tradingview": "TradingView"}.get(meta.get("provider"), meta.get("provider") or "TradingView")
    mode = meta.get("update_mode")
    if mode == "realtime":
        return f"{provider} · {t('realtime')}"
    if mode == "delayed" and meta.get("delay_min"):
        return f"{provider} · {t('delayed')} {meta['delay_min']} {t('min')}"
    if mode == "endofday":
        return f"{provider} · {t('end of day')}"
    if mode == "unknown" and meta.get("raw"):
        return f"{provider} · {t('update mode')}: {meta['raw']}"
    return f"{provider} · {t('delayed ~15 min')}"       # caches saved before metadata existed


def junk_toggle(container) -> None:
    container.toggle(t("Hide junk"), key="hide_junk",
                     help="מסתיר מניות זבל: " + ", ".join(I.ui(label, I.HE) for label, _ in JUNK_RULES)
                          + ". מניה בלי נתון לא מוסתרת")


def nav() -> str:
    st.segmented_control("Page", PAGES, key="page", label_visibility="collapsed", required=True, format_func=t,
                         help="סורק = סריקת מניות, קבוצות = ביצועים לפי סקטור ותעשייה, מפת חום = מפת השוק, "
                              "תיקים ומעקב = רשימות מעקב ותיקי השקעות שלך, מדריך = הסבר על הסקרינר ועל המונחים")
    return ss.page


def toolbar(presets: dict[str, P.Preset]) -> None:
    names = [CUSTOM] + list(presets)
    if ss.get("preset") not in names:
        ss.preset = CUSTOM
    c1, c2, c3, c4, c7, c5, c6 = st.columns([2.5, 2.3, 1.7, 1.3, 1.4, 1.1, 1.1], vertical_alignment="bottom")
    def preset_name(n: str) -> str:
        return t(n) if n == CUSTOM else n if n.startswith(USER_PREFIX) else I.preset(n, L())

    c1.selectbox(t("Screen"), names, key="preset", on_change=on_preset, format_func=preset_name,
                 help="סריקות מוכנות. בחירה בסריקה ממלאת את הפילטרים — אפשר לשנות אותם אחר כך")
    c2.text_input(t("Search"), key="query", placeholder=t("Ticker / company, or list: AAPL, MSFT"),
                  help="חיפוש לפי סימול או שם חברה. אפשר להדביק רשימת סימולים מופרדת בפסיקים")
    sortable = [k for k, c in COLUMNS.items() if c.kind not in ("text", "flag") or k in ("ticker", "company")]
    c3.selectbox(t("Order by"), sortable, key="sort_col", format_func=col_label)
    c4.toggle(t("Ascending"), key="sort_asc")
    junk_toggle(c7)
    c5.button(t("Reset"), on_click=on_reset, use_container_width=True, help="ניקוי כל הפילטרים")
    with c6.popover("", icon=":material/bookmark:", use_container_width=True, help="שמירת הסריקה הנוכחית / מחיקת סריקה שמורה"):
        st.text_input(t("Save current screen as"), key="save_name")
        st.button(t("Save"), on_click=on_save, type="primary")
        user = P.load_user()
        if user:
            st.divider()
            st.selectbox(t("Delete saved screen"), list(user), key="delete_name")
            st.button(t("Delete"), on_click=on_delete)
    note = presets.get(ss.preset).note if ss.preset in presets else ""
    if note:
        hebrew_note(f"ℹ️ {note}")


def filter_panel(df: pd.DataFrame) -> None:
    sel = selections()
    n_active = len(active(sel))
    title = f"{t('Filters')} ({n_active} {t('active')})" if n_active else t("Filters")
    with st.expander(title, expanded=True):
        counts = {g: sum(1 for f in FILTERS if f.group == g and sel[f.key] != ANY) for g in GROUPS}
        tabs = st.tabs([f"{t(g)} ({counts[g]})" if counts[g] else t(g) for g in GROUPS])
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
                        col.selectbox(I.filter_label(f.key, f.label, L()), labels, key=key, help=f.tooltip,
                                      format_func=opt)


def results(df: pd.DataFrame, universe: pd.DataFrame, res: pd.DataFrame, snap,
            prev: history.Previous | None) -> None:
    top = st.columns([3, 5, 1.6], vertical_alignment="center")
    of = f"<small>/ {len(universe):,}</small>" if len(res) != len(universe) else ""
    top[0].markdown(f'<div class="hs-count">{len(res):,} {html.escape(t("results"))}{of}</div>',
                    unsafe_allow_html=True)
    with top[1]:
        st.segmented_control("View", list(VIEWS), key="view", label_visibility="collapsed", required=True,
                             format_func=t)
    view_cols = VIEWS[ss.view]
    chips = described()
    if chips:
        filter_of = {f"{k}: {v}": f for (k, v), f in zip(chips, active(selections()))}   # pill label -> filter
        labels = list(filter_of)
        key = pills_key(labels)
        st.pills(t("Active filters"), options=labels, default=labels, selection_mode="multi",
                 label_visibility="collapsed", key=key, on_change=on_remove_pill, args=(key, filter_of),
                 help="לחיצה על פילטר מבטלת אותו")
    notes = []                                   # end-user explanations are Hebrew (CLAUDE.md)
    if ss.hide_junk and len(universe) < len(df):
        notes.append(f"{len(df) - len(universe):,} מניות זבל מוסתרות ({t('Hide junk')})")
    if prev is not None:
        n_new = int((res["new"] == history.NEW).sum())
        top[2].toggle(f"{t('Only new')} ({n_new:,})", key="only_new",
                      help=f"רק מניות שנכנסו לתוצאות מאז ה-snapshot של {prev.day:%d/%m/%Y}")
        view_cols = [view_cols[0], "new", *view_cols[1:]]
        notes.append(f"NEW = המניה לא הייתה בתוצאות האלה ב-{prev.day:%d/%m/%Y}")
        if ss.get("only_new"):
            res = res[res["new"] == history.NEW]
    if notes:
        hebrew_note(" · ".join(notes))
    if prev is not None and prev.missing:
        hebrew_note(f"ההשוואה ל-{prev.day:%d/%m/%Y} חלקית: ה-snapshot נשמר בגרסה קודמת בלי {len(prev.missing)} "
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
        table = table.style.map(_sign_color, subset=signed)

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
        hebrew_note("👆 לחץ על שורה בטבלה כדי לפתוח גרף ונתונים של המניה")
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


def snapshot_table(row: pd.Series) -> str:
    """Compact Finviz-style key/value table (HTML, no scripts)."""
    end = "left" if L() == I.HE else "right"                 # values sit at the far end of the row
    out = ['<table class="hs-snapshot" style="width:100%;border-collapse:collapse;font-size:13px">']
    for title, keys in DETAIL_FIELDS:
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

    return (f'<div dir="ltr" style="display:flex;align-items:center;gap:6px;font-size:12px;color:{theme.MUTED};'
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
        sector, country = I.value("sector", row["sector"], L()), I.value("country", row["country"], L())
        lrm = "\u200e"     # keeps a trailing "." of English names ("Apple Inc.") in place in RTL
        st.caption(f"{row['company']}{lrm}  \n{sector} · {row['industry']}{lrm} · {country}"
                   + (f"  \n{row['indexes']}" if row.get("indexes") else ""))
        st.metric(t("Price"), fmt_value("price", row["price"]), f"{row['change']:+.2f}%")
        bar = range52_bar(row.get("price"), row.get("low52"), row.get("high52"))
        if bar:
            st.caption(f"**{t('52-Week Range')}**")
            st.html(bar)
        if row.get("signals"):
            st.caption(f"**{t('Signals')}:** {I.signal_tags(row['signals'], L())}")
        st.html(snapshot_table(row))
        st.markdown(f"[{t('Open in TradingView')} ↗]({export.tradingview_url(row['tv_symbol'])}) · "
                    f"[Finviz ↗](https://finviz.com/quote.ashx?t={row['ticker']})")
        add_to_watchlist_button(row["ticker"])


def add_to_watchlist_button(ticker: str) -> None:
    watchlists = [n for n, p in PF.load_all().items() if p.kind == PF.WATCHLIST]
    options = watchlists + [NEW_WATCHLIST]
    if ss.get("add_wl") not in options:
        ss.add_wl = options[0]
    with st.popover(t("Add to watchlist"), icon=":material/playlist_add:", use_container_width=True):
        st.selectbox(t("Watchlist"), options, key="add_wl", format_func=lambda n: t(n) if n == NEW_WATCHLIST else n)
        if ss.add_wl == NEW_WATCHLIST:
            st.text_input(t("List name"), key="add_wl_name")
        st.button(f"{t('Add')} {ticker}", on_click=on_add_to_watchlist, args=(ticker,), type="primary",
                  key="add_wl_btn")


def export_bar(res: pd.DataFrame, view_cols: list[str], snap) -> None:
    st.divider()
    title = ss.preset.removeprefix(USER_PREFIX) if ss.get("preset") not in (None, CUSTOM) else "custom"
    stem = export.file_stem(title)
    c0, c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.2, 1.2], vertical_alignment="center")
    all_cols = c0.toggle(t("Export all columns"), key="export_all",
                         help="כבוי: רק העמודות שבתצוגה הנוכחית. דלוק: כל השדות (כ-110 עמודות; "
                              "תבניות הנרות מסוכמות בעמודת Signals)")
    cols = [k for k, c in COLUMNS.items() if k != "tv_symbol" and c.kind != "flag"] if all_cols else view_cols
    filters_desc = described()
    if ss.get("query"):
        filters_desc = filters_desc + [(t("Search"), ss.query)]
    headers = {k: col_label(k) for k in cols}
    preset = ss.get("preset")
    shown_title = (t("(custom)") if preset in (None, CUSTOM) else title if preset.startswith(USER_PREFIX)
                   else I.preset(title, L()))

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
    hebrew_note(f"עותק של כל קובץ שמורד נשמר גם בתיקייה: {export.EXPORT_DIR}")


# ─────────────────────────── Groups / Heatmap pages ────────────────────────

HEAT_COLORS = {"change": 3, "perf_w": 6, "perf_m": 10, "perf_q": 20, "perf_h": 30, "perf_ytd": 40, "perf_y": 50}
HEAT_UNIVERSES = ["S&P 500", "Nasdaq 100", "Screener results", "All stocks (top 1,000)"]


def groups_page(universe: pd.DataFrame, res: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns([1.6, 1.8, 2.4, 1.6, 1.2], vertical_alignment="bottom")
    ss.setdefault("g_by", "Sector")
    ss.setdefault("g_weight", "Equal")
    c1.segmented_control(t("Group by"), ["Sector", "Industry"], key="g_by", required=True, format_func=t)
    c2.segmented_control(t("Averages"), ["Equal", "Market cap"], key="g_weight", required=True, format_func=t,
                         help="שווה = כל מניה נספרת פעם אחת. שווי שוק = מניות גדולות משפיעות יותר (כמו מדד)")
    base = res if ss.get("g_results") else universe
    by = ss.g_by.lower()
    if by == "industry":
        sectors = ["All sectors"] + sorted(base["sector"].dropna().astype(str).unique())
        if ss.get("g_sector") not in sectors:
            ss.g_sector = "All sectors"
        c3.selectbox(t("Sector"), sectors, key="g_sector", help="הצגת התעשיות של סקטור אחד בלבד",
                     format_func=lambda v: t(v) if v == "All sectors" else I.value("sector", v, L()))
        if ss.g_sector != "All sectors":
            base = base[base["sector"] == ss.g_sector]
    c4.toggle(t("Only screener results"), key="g_results",
              help="דלוק: הקבוצות מחושבות רק מהמניות שעברו את הפילטרים בדף הסורק")
    junk_toggle(c5)
    if base.empty:
        st.info("אין מניות להצגה — הפילטרים או החיפוש לא החזירו תוצאות")
        return
    table = G.group_performance(base, by=by, weight="cap" if ss.g_weight == "Market cap" else "equal")
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
    styled = table.style.map(_sign_color, subset=G.PERF)
    st.dataframe(styled, hide_index=True, height=min(38 + 35 * len(table), 760), column_config=config,
                 key="groups_table")
    hebrew_note("ביצועים ממוצעים של המניות בכל קבוצה. לחיצה על כותרת עמודה ממיינת. "
                "מומלץ להשאיר את הסתרת מניות הזבל דלוקה — מניות זבל עם תנועות של מאות אחוזים "
                "מעוותות ממוצע שווה-משקל.")


def heatmap_page(universe: pd.DataFrame, res: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([2.4, 2.4, 1.2], vertical_alignment="bottom")
    ss.setdefault("h_universe", HEAT_UNIVERSES[0])
    ss.setdefault("h_color", "change")
    c1.selectbox(t("Stocks"), HEAT_UNIVERSES, key="h_universe", format_func=t,
                 help="אילו מניות להציג. S&P 500 מקביל למפת השוק של Finviz")
    c2.selectbox(t("Color by"), list(HEAT_COLORS), key="h_color", format_func=col_label,
                 help="ירוק = עלייה, אדום = ירידה. הצבע רווי בקצוות הסקאלה (למשל ±3% לשינוי יומי)")
    junk_toggle(c3)
    match ss.h_universe:
        case "Screener results":
            data = res
        case "All stocks (top 1,000)":
            data = universe.nlargest(1000, "market_cap")
        case label:
            data = universe[in_index(label)(universe)]
    if data.empty:
        st.info("אין מניות להצגה")
        return
    color = ss.h_color
    fig = G.heatmap_figure(display_values(data), color, limit=HEAT_COLORS[color], root=t("US stocks"),
                           stocks=t("stocks"), cap_label=t("Market cap"))
    st.plotly_chart(fig, key="heatmap", config={"displaylogo": False})
    hebrew_note(f"{len(data):,} מניות. גודל = שווי שוק, צבע = {col_label(color)}. "
                "לחיצה על סקטור או תעשייה מגדילה אותם; לחזרה לחץ על שורת הניווט למעלה.")


# ─────────────────────────── Portfolios page ───────────────────────────────

NEW_WATCHLIST = "— New watchlist —"


def _by_ticker(df: pd.DataFrame, col: str = "price") -> pd.Series:
    """ticker -> a universe column (prices come from the screener's data: no extra requests)."""
    return df.drop_duplicates("ticker").set_index("ticker")[col]


def _money(v: float) -> str:
    return "—" if pd.isna(v) else f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def on_pf_import():
    imp, name = ss.get("pf_pending"), (ss.get("pf_new_name") or "").strip()
    if imp is None or not imp.ok or not name:
        return
    try:
        PF.save(PF.from_imported(name, imp, _by_ticker(ss.df)))
    except Exception as e:
        log.warning("cannot save portfolio %r: %s", name, e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    ss.pf_name, ss.pf_new_name = name, ""
    ss.pf_upload = ss.get("pf_upload", 0) + 1          # a new uploader widget: the file is cleared
    st.toast(f"הרשימה נשמרה: “{name}”", icon="💾")


def on_pf_new_watchlist():
    name = (ss.get("pf_new_name") or "").strip()
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load lists: %s", e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    if not name or name in lists:
        return
    try:
        PF.save(PF.PList(name, PF.WATCHLIST))
    except Exception as e:
        log.warning("cannot save watchlist %r: %s", name, e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    ss.pf_name, ss.pf_new_name = name, ""


def add_tickers(plist: PF.PList, text: str, prices: pd.Series) -> tuple[list[str], list[str]]:
    """'aapl, msft nvda' -> (added, not in the screener). Tickers already in the list are skipped."""
    added, unknown = [], []
    for tk in filter(None, (PF.normalize_ticker(x) for x in re.split(r"[,;\s]+", text))):
        if tk not in prices.index:
            unknown.append(tk)
        elif PF.add_to_watchlist(plist, tk, prices[tk]):
            added.append(tk)
    return added, unknown


def _add_and_report(plist: PF.PList, text: str) -> None:
    added, unknown = add_tickers(plist, text, _by_ticker(ss.df))
    try:
        PF.save(plist)
    except Exception as e:
        log.warning("cannot save list %r: %s", plist.name, e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    if added:
        st.toast(f"נוספו ל“{plist.name}”: {', '.join(added)}", icon="✅")
    if unknown:
        st.toast(f"לא נמצאו בסקרינר: {', '.join(unknown)}", icon="⚠️")


def on_pf_add_tickers():
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load list: %s", e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    name = ss.get("pf_name")
    if not name or name not in lists:
        st.toast("הרשימה נמחקה בחלון אחר", icon="⚠️")
        return
    plist = lists[name]
    if plist is not None:
        _add_and_report(plist, ss.get("pf_add") or "")
    ss.pf_add = ""


def on_add_to_watchlist(ticker: str):
    """From the stock details panel."""
    name = ss.get("add_wl")
    if name == NEW_WATCHLIST:
        name = (ss.get("add_wl_name") or "").strip()
        if not name:
            return
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load lists: %s", e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    plist = lists.get(name) or PF.PList(name, PF.WATCHLIST)
    if plist.kind != PF.WATCHLIST:
        st.toast(f"“{name}” הוא תיק, לא רשימת מעקב", icon="⚠️")
        return
    _add_and_report(plist, ticker)
    ss.add_wl, ss.add_wl_name = name, ""


def on_pf_remove():
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load list: %s", e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    name = ss.get("pf_name")
    if not name or name not in lists:
        st.toast("הרשימה נמחקה בחלון אחר", icon="⚠️")
        return
    plist = lists[name]
    if plist is not None:
        plist.rows = [r for r in plist.rows if r["ticker"] != ss.get("pf_remove")]
        try:
            PF.save(plist)
        except Exception as e:
            log.warning("cannot save list: %s", e)
            st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
            return


def on_pf_trade():
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load list: %s", e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    name = ss.get("pf_name")
    if not name or name not in lists:
        st.toast("הרשימה נמחקה בחלון אחר", icon="⚠️")
        return
    plist = lists[name]
    if plist is None:
        return
    raw = pd.DataFrame([{"ticker": ss.get("pf_t_ticker") or "", "date": ss.get("pf_t_date"),
                         "price": ss.get("pf_t_price"), "quantity": ss.get("pf_t_qty"),
                         "side": ss.get("pf_t_side"), "fees": ss.get("pf_t_fees") or 0.0,
                         "note": ss.get("pf_t_note") or ""}])
    imp = PF.parse(raw)
    errors = [I.problem(PF.Problem(0, p.code, p.value)) for p in imp.problems]
    if not errors:
        errors = [I.problem(PF.Problem(0, "oversold", tk)) for tk in PF.add_trades(plist, imp.rows)]
    if errors:
        ss.pf_trade_errors = errors
        return
    try:
        PF.save(plist)
    except Exception as e:
        log.warning("cannot save trade: %s", e)
        st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
        return
    ss.pf_trade_errors = []
    for k in ("pf_t_ticker", "pf_t_price", "pf_t_qty", "pf_t_fees", "pf_t_note"):
        ss.pop(k, None)
    st.toast(f"העסקה נוספה: {imp.rows['ticker'].iloc[0]}", icon="✅")


def on_pf_delete():
    name = ss.get("pf_name")
    if name:
        try:
            PF.delete(name)
        except Exception as e:
            log.warning("cannot delete portfolio %r: %s", name, e)
            st.toast("לא ניתן לשמור כרגע, נסו שוב בעוד רגע", icon="⚠️")
            return
        ss.pop("pf_name", None)


def on_show_in_screener(tickers: list[str]):
    on_reset()                                    # filters would hide part of the list
    ss.query = ", ".join(tickers)
    ss.page = "Screener"


def import_panel(lists: dict[str, PF.PList], df: pd.DataFrame) -> None:
    with st.expander(t("New list / import a file"), expanded=not lists, icon=":material/upload_file:"):
        hebrew_note("קובץ CSV או אקסל, שורה לכל עסקה: סימול, תאריך, שער, כמות, ואפשר גם פעולה (קנייה/מכירה), "
                    "עמלות והערה. קובץ עם עמודת סימולים בלבד נשמר כרשימת מעקב. הכותרות יכולות להיות בעברית. "
                    "כדי לערוך רשימה קיימת: מייצאים אותה, משנים באקסל ומייבאים שוב באותו שם.")
        c1, c2 = st.columns([1, 1.4], vertical_alignment="bottom")
        name = c1.text_input(t("List name"), key="pf_new_name").strip()
        up = c2.file_uploader(t("CSV or Excel file"), type=["csv", "xlsx", "xls"],
                              key=f"pf_file_{ss.get('pf_upload', 0)}")
        ss.pf_pending = None
        if up is not None:
            try:
                imp = PF.parse(PF.read_table(up.getvalue(), up.name))
            except Exception as e:
                log.warning("cannot read %s: %s", up.name, e)
                st.error("לא ניתן לקרוא את הקובץ. שמור אותו באקסל כ-CSV או כ-xlsx ונסה שוב.")
                imp = None
            if imp is not None and imp.problems:
                st.error("יש לתקן בקובץ ולטעון שוב:\n\n" + "\n".join(f"- {I.problem(p)}" for p in imp.problems[:15])
                         + (f"\n- ועוד {len(imp.problems) - 15}" if len(imp.problems) > 15 else ""))
            elif imp is not None and imp.rows.empty:
                st.error("הקובץ ריק")
            elif imp is not None:
                ss.pf_pending = imp
                n_tk = imp.rows["ticker"].nunique()
                kind_str = t("Portfolio") if imp.kind == PF.PORTFOLIO else t("Watchlist")
                details = [f"סוג: **{kind_str}**"]
                if imp.kind == PF.PORTFOLIO:
                    details.append(f"פורמט תאריך: **{imp.date_format}**")
                    dates = imp.rows["date"].dropna()
                    if not dates.empty:
                        d_min, d_max = dates.min(), dates.max()
                        details.append(f"טווח תאריכים: **{d_min:%d/%m/%Y} עד {d_max:%d/%m/%Y}**")
                    details.append(f"עסקאות: **{len(imp.rows)}** ב-**{n_tk}** מניות")
                else:
                    details.append(f"מניות: **{n_tk}**")

                st.success(" · ".join(details))
                st.caption("תצוגה מקדימה (5 שורות ראשונות):")
                preview_cols = [c for c in PF.COLUMNS if c in imp.rows.columns and not imp.rows[c].isna().all()]
                st.dataframe(imp.rows[preview_cols].head(5), hide_index=True)

                unknown = sorted(set(imp.rows["ticker"]) - set(df["ticker"]))
                if unknown:
                    hebrew_note(f"אין בסקרינר מחיר ל: {', '.join(unknown)} (למשל קרנות סל). "
                                "הן יישמרו, אבל בלי שווי נוכחי.")
        if name and name in lists:
            hebrew_note(f"כבר יש רשימה בשם “{name}”. שמירה עם קובץ תחליף אותה.")
        b1, b2, b3 = st.columns([1, 1, 1])
        b1.button(t("Save list"), on_click=on_pf_import, type="primary", use_container_width=True,
                  disabled=not (name and ss.pf_pending is not None), key="pf_save")
        b2.button(t("Empty watchlist"), on_click=on_pf_new_watchlist, use_container_width=True,
                  disabled=not name or name in lists, key="pf_empty", icon=":material/add:")
        b3.download_button(t("Sample file"), PF.template_csv(), "halgom_portfolio_sample.csv", mime="text/csv",
                           on_click="ignore", use_container_width=True, icon=":material/download:")


def watchlist_view(plist: PF.PList, df: pd.DataFrame) -> None:
    w = PF.watch_performance(plist.frame(), _by_ticker(df))
    if w.empty:
        hebrew_note("הרשימה ריקה. מוסיפים סימולים בתיבה למטה, או מכפתור “הוספה לרשימת מעקב” בפרטי מניה בסורק.")
    else:
        w["company"] = w["ticker"].map(_by_ticker(df, "company"))
        w["change"] = w["ticker"].map(_by_ticker(df, "change"))
        cols = ["ticker", "company", "date", "price", "current", "change", "since_added_pct", "note"]
        cc = st.column_config
        config = {"ticker": cc.TextColumn(t("Ticker"), pinned=True, width="small"),
                  "company": cc.TextColumn(col_label("company"), width="medium"),
                  "date": cc.DateColumn(t("Added"), format="DD/MM/YYYY"),
                  "price": cc.NumberColumn(t("Price then"), format="%.2f"),
                  "current": cc.NumberColumn(t("Price"), format="%.2f"),
                  "change": cc.NumberColumn(col_label("change"), format="%.2f%%"),
                  "since_added_pct": cc.NumberColumn(t("Since added"), format="%.2f%%"),
                  "note": cc.TextColumn(t("Notes"))}
        st.dataframe(w[cols].style.map(_sign_color, subset=["change", "since_added_pct"]), hide_index=True,
                     column_config=config, placeholder="—", key="pf_watch_table", height=min(38 + 35 * len(w), 600))
    c1, c2, c3 = st.columns([2.4, 1, 1.4], vertical_alignment="bottom")
    c1.text_input(t("Add tickers"), key="pf_add", placeholder="AAPL, MSFT")
    c2.button(t("Add"), on_click=on_pf_add_tickers, use_container_width=True, key="pf_add_btn")
    if not w.empty:
        with c3.popover(t("Remove ticker"), icon=":material/remove:", use_container_width=True):
            st.selectbox(t("Ticker"), list(w["ticker"]), key="pf_remove")
            st.button(t("Remove"), on_click=on_pf_remove, key="pf_remove_btn")


def portfolio_view(plist: PF.PList, df: pd.DataFrame) -> None:
    trades = plist.frame()
    val = PF.valuation(PF.positions(trades), _by_ticker(df))
    val["company"] = val["ticker"].map(_by_ticker(df, "company"))
    val["change"] = val["ticker"].map(_by_ticker(df, "change"))
    held = val[val["quantity"] > 0]
    priced = held[held["price"].notna()]
    value, cost = priced["value"].sum(), priced["cost"].sum()
    today = (priced["value"] * priced["change"] / (100 + priced["change"])).sum()
    cc = st.column_config
    m = st.columns(4)
    m[0].metric(t("Market value"), _money(value), f"{_money(today)} {t('today')}" if len(priced) else None)
    m[1].metric(t("Cost"), _money(cost))
    m[2].metric(t("Unrealized P&L"), _money(value - cost),
                f"{(value - cost) / cost * 100:+.2f}%" if cost > 0 else None)
    m[3].metric(t("Realized P&L"), _money(val["realized"].sum()))
    missing = held.loc[held["price"].isna(), "ticker"].tolist()
    if missing:
        hebrew_note(f"אין בסקרינר מחיר ל: {', '.join(missing)} (למשל קרנות סל). הן לא נכללות בשווי ובסיכומים.")
    if held.empty:
        hebrew_note("אין כרגע פוזיציות פתוחות בתיק.")
    else:
        cols = ["ticker", "company", "quantity", "avg_cost", "price", "change", "value", "pnl", "pnl_pct",
                "weight", "annual_pct", "first_buy"]
        config = {"ticker": cc.TextColumn(t("Ticker"), pinned=True, width="small"),
                  "company": cc.TextColumn(col_label("company"), width="medium"),
                  "quantity": cc.NumberColumn(t("Quantity"), format="localized"),
                  "avg_cost": cc.NumberColumn(t("Avg cost"), format="%.2f"),
                  "price": cc.NumberColumn(t("Price"), format="%.2f"),
                  "change": cc.NumberColumn(col_label("change"), format="%.2f%%"),
                  "value": cc.NumberColumn(t("Market value"), format="dollar"),
                  "pnl": cc.NumberColumn(t("P&L"), format="dollar"),
                  "pnl_pct": cc.NumberColumn(t("P&L %"), format="%.2f%%"),
                  "weight": cc.ProgressColumn(t("Weight"), min_value=0, max_value=100, format="%.1f%%"),
                  "annual_pct": cc.NumberColumn(t("Annual return"), format="%.2f%%",
                                                help="תשואה שנתית ממוצעת לפי תאריך הקנייה הממוצע. מוצגת רק "
                                                     "אחרי 90 יום, כי לפני כן המספר מטעה"),
                  "first_buy": cc.DateColumn(t("First buy"), format="DD/MM/YYYY")}
        st.dataframe(held[cols].style.map(_sign_color, subset=["change", "pnl", "pnl_pct", "annual_pct"]),
                     hide_index=True, column_config=config, placeholder="—", key="pf_positions",
                     height=min(38 + 35 * len(held), 600))
    hebrew_note("עלות ממוצעת כוללת עמלות. ספליטים ודיבידנדים אחרי הקנייה עוד לא מחושבים.")
    with st.expander(f"{t('Trades')} ({len(trades)})"):
        shown = trades.assign(side=trades["side"].map(t))
        st.dataframe(shown, hide_index=True, placeholder="—", key="pf_trades", column_config={
            "ticker": t("Ticker"), "date": cc.DateColumn(t("Date"), format="DD/MM/YYYY"),
            "price": cc.NumberColumn(t("Price"), format="%.2f"), "quantity": t("Quantity"),
            "side": t("Side"), "fees": cc.NumberColumn(t("Fees"), format="%.2f"), "note": t("Notes")})
    with st.popover(t("Add trade"), icon=":material/add:"):
        with st.form("pf_trade", border=False):
            c1, c2 = st.columns(2)
            c1.text_input(t("Ticker"), key="pf_t_ticker")
            c2.date_input(t("Date"), key="pf_t_date", value=dt.date.today(), max_value=dt.date.today(),
                          format="DD/MM/YYYY")
            c1.number_input(t("Price"), key="pf_t_price", min_value=0.0, value=None, step=0.01, format="%.2f")
            c2.number_input(t("Quantity"), key="pf_t_qty", min_value=0.0, value=None, step=1.0)
            c1.radio(t("Side"), ["buy", "sell"], key="pf_t_side", format_func=t, horizontal=True)
            c2.number_input(t("Fees"), key="pf_t_fees", min_value=0.0, value=0.0, step=0.5)
            st.text_input(t("Notes"), key="pf_t_note")
            st.form_submit_button(t("Add trade"), on_click=on_pf_trade, type="primary")
        for e in ss.get("pf_trade_errors") or []:
            st.error(e)


def portfolios_page(df: pd.DataFrame) -> None:
    lists = PF.load_all()
    import_panel(lists, df)
    if not lists:
        hebrew_note("עוד אין רשימות. אפשר לייבא קובץ מהברוקר או מאקסל, ליצור רשימת מעקב ריקה, "
                    "או להוסיף מניה מפרטי המניה בסורק.")
        return
    names = list(lists)
    if ss.get("pf_name") not in names:
        ss.pf_name = names[0]
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2], vertical_alignment="bottom")
    c1.selectbox(t("List"), names, key="pf_name",
                 format_func=lambda n: f"{n} · {t('Portfolio' if lists[n].kind == PF.PORTFOLIO else 'Watchlist')}")
    plist = lists[ss.pf_name]
    tickers = list(dict.fromkeys(r["ticker"] for r in plist.rows))
    c2.button(t("Show in screener"), on_click=on_show_in_screener, args=(tickers,), disabled=not tickers,
              use_container_width=True, icon=":material/search:", key="pf_show",
              help="פותח את הסורק עם המניות של הרשימה בלבד (הפילטרים מתאפסים)")
    c3.download_button(t("Export list"), PF.to_csv(plist), f"{export.file_stem(plist.name)}.csv", mime="text/csv",
                       on_click="ignore", use_container_width=True, icon=":material/download:",
                       help="קובץ CSV באותו מבנה של הייבוא: גיבוי, עריכה באקסל או העברה למחשב אחר")
    with c4.popover(t("Delete list"), icon=":material/delete:", use_container_width=True):
        hebrew_note(f"למחוק את “{plist.name}”? אי אפשר לבטל. כדאי לייצא קודם.")
        st.button(t("Delete"), on_click=on_pf_delete, type="primary", key="pf_delete")
    if plist.kind == PF.PORTFOLIO:
        portfolio_view(plist, df)
    else:
        watchlist_view(plist, df)
    hebrew_note("הרשימות נשמרות במחשב הזה בלבד. מחירים מהסקרינר (בעיכוב של כ-15 דקות).")


# ─────────────────────────── page ──────────────────────────────────────────

def goodbye() -> None:
    """Shown after Exit: the server stops a moment later, so this is the last thing the tab shows."""
    st.html('<div class="hs-brand" style="justify-content:center;margin-top:18vh"><i></i><b>HalgOM</b>'
            '<span>Screener</span></div>'
            '<div dir="rtl" style="text-align:center;margin-top:1.2rem;font-size:1.4rem;font-weight:600">'
            'הסקרינר נסגר</div>'
            f'<div dir="rtl" style="text-align:center;margin-top:0.6rem;color:{theme.MUTED}">'
            'אפשר לסגור את הלשונית הזו. כדי לפתוח שוב — לחיצה כפולה על הקיצור "HalgOM Screener" '
            'בשולחן העבודה.</div>')
    if not ss.get("exit_sent"):
        ss.exit_sent = True
        lifecycle.stop_server("Exit button")


def main() -> None:
    _setup_logging()
    ss.setdefault("lang", I.HE)
    ss.setdefault("large_text", False)
    css = theme.CSS + (theme.RTL_CSS if L() == I.HE else "") + (theme.LARGE_TEXT_CSS if ss.large_text else "")
    st.html(css)
    if ss.get("exiting"):
        goodbye()
        return
    _watchdog()
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
    universe = hide_junk(df, ss.hide_junk)       # once per run, shared by every page
    res = search(apply_filters(universe, sel), ss.get("query", ""))
    if page == "Guide":
        st.html(GUIDE.read_text(encoding="utf-8"))      # Hebrew only, like all end-user explanations
    elif page == "Groups":
        groups_page(universe, res)
    elif page == "Heatmap":
        heatmap_page(universe, res)
    elif page == "Portfolios":
        portfolios_page(df)
    else:
        prev = get_previous(snap)                # "New since yesterday" is only shown here
        res = res.assign(new=new_column(res, prev, sel))
        res = res.sort_values(ss.sort_col, ascending=ss.sort_asc, na_position="last")
        results(df, universe, res, snap, prev)
    hebrew_note("נתונים: TradingView (בעיכוב של כ-15 דקות). הכלי לצורכי מחקר בלבד ואינו ייעוץ השקעות.")


main()
