"""Shared UI state and helpers: session state, translation, data loading, the header, navigation, formatting.

Moved verbatim from screener/app.py; behavior unchanged."""
from __future__ import annotations

import datetime as dt
import html
import logging
import logging.handlers
from pathlib import Path

import pandas as pd
import streamlit as st

from screener import history, i18n as I, lifecycle, presets as P, theme
from screener.data import FETCH_LOCK, Store, guard, load_universe
# The provider functions and the data folder are read through their modules at call time (store.DATA_DIR,
# tv.fetch_universe): this module is imported once, while app.py used to be re-executed on every rerun, and
# tests point them elsewhere (CLAUDE.md: paths resolved at call time).
from screener.data import store as store_mod, tradingview as tv
from screener.data.store import market_date, session_date
from screener.derive import add_derived
from screener.fields import COLUMNS, provider_columns
from screener.filters import ANY, FILTERS, JUNK_RULES, active
from screener.market import compute_regime, trend_strength


VERSION = "0.2.0"
MAX_AGE = dt.timedelta(minutes=15)       # TradingView data is ~15 min delayed anyway
RETRY_AFTER = dt.timedelta(minutes=2)    # when offline, don't hammer the network on every click
RETRY_TIMEOUT = 20                         # seconds, for those retries (a normal fetch takes ~15 s)
REFRESH_MIN = dt.timedelta(seconds=60)   # "Refresh data" clicked again sooner: no new request
CUSTOM = "— Custom —"
USER_PREFIX = "★ "
PAGES = ["Screener", "Groups", "Heatmap", "Portfolios", "Guide"]
# provider columns a cached snapshot must have; an older cache without them is refetched
REQUIRED = provider_columns()
# widget keys that must survive while their widget is not on the current page
PERSIST = ["lang", "preset", "query", "sort_col", "sort_asc", "view", "hide_junk", "only_new", "export_all",
           "g_by", "g_weight", "g_sector", "g_results", "h_universe", "h_color", "large_text", "pf_name"]
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
def setup_logging() -> Path:
    log_dir = store_mod.DATA_DIR / "logs"
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
def start_watchdog():
    """One per server: stops it after 15 minutes with no open browser tab (lifecycle.py)."""
    return lifecycle.ensure_idle_watchdog_started()


@st.cache_resource
def get_store() -> Store:
    return Store(store_mod.DATA_DIR)


@st.cache_resource
def _market_holder() -> dict[str, pd.DataFrame | None]:
    return {"last_df": None}


@st.cache_data(ttl=MAX_AGE, show_spinner=False)
def _fetch_market():
    with FETCH_LOCK:
        return guard.call("tradingview_market", tv.fetch_market)


def _market():
    holder = _market_holder()
    try:
        m = _fetch_market()
        if m is not None:
            holder["last_df"] = m
        return m, compute_regime(m)
    except guard.Wait:
        last = holder.get("last_df")
        if last is not None:
            return last, compute_regime(last)
        return None, None
    except Exception:
        log.exception("market fetch failed")
        last = holder.get("last_df")
        if last is not None:
            return last, compute_regime(last)
        return None, None


_market.clear = _fetch_market.clear


def _fetch_universe(**kwargs):
    fn = tv.fetch_universe
    try:
        return guard.call("tradingview", fn, **kwargs)
    except TypeError:
        if kwargs:
            return guard.call("tradingview", fn)
        raise


def get_data(force: bool = False):
    snap = ss.get("snap")
    st_tv = guard.status("tradingview")
    waiting = st_tv.get("waiting", False)
    due = (force or snap is None
           or (snap.source == "stale" and not waiting)
           or (snap.source != "stale" and not dt.timedelta(0) <= snap.age <= MAX_AGE))   # < 0: clock went back
    if due:
        with st.spinner(t("Loading market data from TradingView… (≈15 sec)")):
            try:
                retry = snap is not None and snap.source == "stale"     # offline: don't block a click for 60 s
                snap = load_universe(get_store(), _fetch_universe, MAX_AGE, force=force, required=REQUIRED,
                                     timeout=RETRY_TIMEOUT if retry else None)
            except guard.Wait as w:
                st.error(guard.wait_message(w))
                if st.button(t("Try again"), type="primary"):
                    st.rerun()
                st.stop()
            except Exception as e:
                log.exception("no data available")
                st.error("לא ניתן להוריד נתונים, ואין נתונים שמורים מהפעם הקודמת. "
                         "בדוק את החיבור לאינטרנט ונסה שוב.")
                st.caption(f"{t('Details')}: {e}")
                if st.button(t("Try again"), type="primary"):
                    st.rerun()
                st.stop()
        ss["snap"] = snap
        ss["df"] = add_derived(snap.df, today=market_date(snap.fetched_at))
    return ss.get("snap"), ss.get("df")


def get_previous(snap) -> history.Previous | None:
    """Previous day's snapshot (derived), loaded once per snapshot day."""
    day = session_date(snap.fetched_at)
    if ss.get("prev_for") != day:
        ss["prev"] = history.load_previous(get_store(), day)
        ss["prev_for"] = day
        ss["prev_hits"] = {}
    return ss.get("prev")


def new_column(res: pd.DataFrame, prev: history.Previous | None, sel: dict[str, str]) -> pd.Series:
    """'NEW' tags; yesterday's matches are computed once per (filters, junk switch), not on every click."""
    if prev is None:
        return history.mark_new(res, None)
    key = (tuple(sorted(active(sel).items())), ss.get("hide_junk"))
    if key not in ss.get("prev_hits"):
        ss.get("prev_hits")[key] = history.previous_tickers(prev, sel, ss.get("hide_junk"))
    return history.mark_new(res, ss.get("prev_hits")[key])


def all_presets() -> dict[str, P.Preset]:
    out = dict(P.BUILTIN)
    out.update({USER_PREFIX + n: p for n, p in P.load_user().items()})
    return out


def selections() -> dict[str, str]:
    return {f.key: ss.get(f"f_{f.key}", ANY) for f in FILTERS}


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


def on_reset():
    for f in FILTERS:
        ss[f"f_{f.key}"] = ANY
    ss["preset"] = CUSTOM
    ss["query"] = ""
    ss["only_new"] = False


def preset_modified(presets: dict[str, P.Preset]) -> bool:
    """The selected preset's filters were changed by hand since it was chosen."""
    p = presets.get(ss.get("preset"))
    return p is not None and active(selections()) != active({f.key: p.filters.get(f.key, ANY) for f in FILTERS})


def on_exit():
    ss["exiting"] = True


# ─────────────────────────── rendering helpers ─────────────────────────────

def column_config(key: str):
    c = COLUMNS[key]
    cc = st.column_config
    kw = {"label": col_label(key), "help": c.help}
    match c.kind:
        case "text":
            width_map = {
                "signals": 300, "indexes": 140, "new": 60,
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


def sign_color(v) -> str:
    if isinstance(v, (int, float)) and not pd.isna(v) and v != 0:
        return f"color: {theme.GAIN}" if v > 0 else f"color: {theme.LOSS}"
    return ""


def trend_text(label: str) -> str:
    """'Trend strong ▲' -> 'מגמה חזקה ▲' (the arrow is kept)."""
    arrow = "▲" if "▲" in label else "▼" if "▼" in label else ""
    return f"{t(label.replace(arrow, '').strip() if arrow else label)} {arrow}".strip()


def header(snap, n_universe: int) -> None:
    with st.container(key="masthead"):
        _header(snap, n_universe)


def _header(snap, n_universe: int) -> None:
    left, mid, right = st.columns([2.5, 4.8, 2.7], vertical_alignment="center")
    with left:
        st.html('<div class="hs-brand"><i></i><b>HalgOM</b><span>Screener</span></div>')
        st.caption(f"v{VERSION} · {t('US stocks')} · {n_universe:,} {t('in universe')}")
        c1, c2 = st.columns([1.3, 1.0], vertical_alignment="center")
        c1.segmented_control("Language", list(I.LANGS), key="lang", format_func=I.LANGS.get, required=True,
                             label_visibility="collapsed")
        c2.toggle(t("Large text"), key="large_text")
    with mid:
        market, regime = _market()
        if market is None:
            st.caption(t("Market data unavailable"))
        else:
            cols = st.container(key="tape").columns([1.2, 1, 1, 1, 1])
            colors = {"bull": "green", "bear": "red", "sideways": "orange", "unknown": "gray"}
            with cols[0]:
                st.badge(t(regime.label.upper()), color=colors[regime.label], icon=":material/monitoring:",
                         help="מצב השוק לפי כללי HalgOM:\n\n" + "\n".join(f"- {I.regime_reason(r, L())}" for r in regime.reasons))
                spy = market.loc["SPY"]
                trend = trend_strength(spy.get("adx", float("nan")), spy.get("di_plus", float("nan")),
                                       spy.get("di_minus", float("nan")))
                st.badge(trend_text(trend.label), color=trend.color, icon=":material/trending_up:", help=trend.help)
                st.caption(t("Market regime · SPY ADX"))
            for c, name in zip(cols[1:], ["SPY", "QQQ", "VIX", "HYG"]):
                if name not in market.index:
                    c.metric(name, "—", None)
                    continue
                row = market.loc[name]
                close_val = "—" if pd.isna(row.get("close")) else f"{row.close:,.2f}"
                change_val = None if pd.isna(row.get("change")) else f"{row.change:+.2f}%"
                sma20_str = "—" if pd.isna(row.get("sma20")) else f"{row.sma20:,.2f}"
                sma50_str = "—" if pd.isna(row.get("sma50")) else f"{row.sma50:,.2f}"
                sma200_str = "—" if pd.isna(row.get("sma200")) else f"{row.sma200:,.2f}"
                vs200 = "" if pd.isna(row.get("sma200")) or pd.isna(row.get("close")) or name == "VIX" else (
                    " · \u200e▲SMA200" if row.close > row.sma200 else " · \u200e▼SMA200")   # LRM: arrow stays left in RTL
                c.metric(name, close_val, change_val,
                         delta_color="inverse" if name == "VIX" else "normal",
                         help=f"{name} vs SMA20/50/200: {sma20_str} / {sma50_str} / {sma200_str}")
                if vs200:
                    c.caption(vs200.removeprefix(" · "))
    with right:
        src = t({"live": "live", "cache": "cached", "stale": "OFFLINE"}[snap.source])
        st.caption(f"{t('Data time')}: **{snap.fetched_at:%H:%M}** · {snap.fetched_at:%d/%m/%Y} ({src}) · "
                   f"{source_text(snap.meta)}",
                   help="מקור הנתונים ומצב העדכון כפי שהספק מדווח. delayed = בעיכוב, realtime = בזמן אמת")
        r1, r2 = st.columns([1.4, 1.3])
        if r1.button(t("Refresh data"), icon=":material/refresh:", use_container_width=True):
            if snap.source != "stale" and snap.age < REFRESH_MIN:      # every click is a TradingView request
                st.toast(f"{t('Data was updated')} {max(0, int(snap.age.total_seconds()))} {t('seconds ago')}", icon="ℹ️")
            else:
                if snap.fetched_at is not None:
                    with guard._STATE_LOCK:
                        st_data = guard._load_state()
                        pstate = st_data.get("tradingview", {})
                        lc = guard._parse_dt(pstate.get("last_call"))
                        s_dt = snap.fetched_at.astimezone() if snap.fetched_at.tzinfo else snap.fetched_at.astimezone()
                        if lc is not None and s_dt < lc:
                            pstate["last_call"] = guard._format_dt(s_dt)
                            st_data["tradingview"] = pstate
                            guard._save_state(st_data)
                w = guard.check_wait("tradingview")
                if w is not None:
                    st.toast(guard.wait_message(w), icon="⚠️")
                else:
                    get_data(force=True)
                    _fetch_market.clear()
                    st.rerun()
        with r2.popover(t("Exit"), icon=":material/power_settings_new:", use_container_width=True,
                        help="סגירת הסקרינר"):
            hebrew_note(t("Closes the screener and the black window. Next time, open again from the desktop shortcut."))
            st.button(t("Close the screener"), key="exit", type="primary", on_click=on_exit)
    if snap.source == "stale":
        msg = snap.error or f"{t('Cannot connect to TradingView right now - showing saved data from')} {snap.fetched_at:%d/%m/%Y} {t('at')} {snap.fetched_at:%H:%M}. {t('Will try again automatically in a few minutes.')}"
        st.warning(f"⚠️ {msg}")


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
    return ss.get("page")


# ─────────────────────────── page ──────────────────────────────────────────

def goodbye() -> None:
    """Shown after Exit: the server stops a moment later, so this is the last thing the tab shows."""
    st.html('<div class="hs-brand" style="justify-content:center;margin-top:18vh"><i></i><b>HalgOM</b>'
            '<span>Screener</span></div>'
            '<div dir="rtl" style="text-align:center;margin-top:1.2rem;font-size:1.4rem;font-weight:600">'
            'הסקרינר נסגר</div>'
            f'<div dir="rtl" style="text-align:center;margin-top:0.6rem;color:{theme.MUTED}">'
            'אפשר לסגור את הלשונית הזו, וגם לשוניות אחרות של הסקרינר אם פתוחות. כדי לפתוח שוב — '
            'לחיצה כפולה על הקיצור "HalgOM Screener" בשולחן העבודה.</div>')
    if not ss.get("exit_sent"):
        ss["exit_sent"] = True
        lifecycle.stop_server("Exit button")
