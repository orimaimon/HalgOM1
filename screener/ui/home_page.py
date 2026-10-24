"""
"הבוקר שלי" (My Morning) home page UI component.

Displays a daily digest for the user:
- New stocks matching saved & built-in screens since yesterday.
- Biggest daily price movers in the user's portfolios and watchlists.
- Upcoming earnings in the next 14 days for held positions.
- Today's fired alerts.
- Plain text preview of the daily report.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from screener import daily_report, home, i18n, portfolio, presets, theme
from screener.data.store import Store, session_date
from screener.ui.common import L, col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def _file_mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


def _snapshot_state_key(store: Store) -> tuple:
    stamps = []
    for d in store.snapshot_dates():
        p = store.snapshot_path(d)
        if p.exists():
            stamps.append((d.isoformat(), p.stat().st_mtime_ns))
    return tuple(stamps)


def _get_cache_key(store: Store, today: dt.date, fired: list | None = None) -> tuple:
    snaps_key = _snapshot_state_key(store)
    p_mtime = _file_mtime(presets.PRESETS_PATH)
    pf_mtime = _file_mtime(portfolio.PORTFOLIOS_PATH)
    fired_key = tuple(fired) if fired else ()
    return (snaps_key, p_mtime, pf_mtime, today.isoformat(), fired_key)


@st.cache_data(max_entries=3)
def _cached_morning(
    cache_key: tuple,
    _universe: pd.DataFrame,
    _store: Store,
    today: dt.date,
    _lists: dict,
    _fired: list | None = None,
) -> home.Morning:
    return home.morning(_universe, _store, today, _lists, fired=_fired)


def home_page(
    universe: pd.DataFrame,
    store: Store,
    regime=None,
    fired: list | None = None,
) -> None:
    """Render the 'הבוקר שלי' (My Morning) home page component."""
    try:
        _render_home_page(universe, store, regime, fired)
    except Exception as e:
        log.exception("Error rendering home page: %s", e)
        hebrew_note(t("Could not load My Morning data right now."))


def _sign_color(val):
    if isinstance(val, (int, float)) and pd.notna(val) and val != 0:
        return f"color: {theme.GAIN}" if val > 0 else f"color: {theme.LOSS}"
    return ""


def _render_home_page(
    universe: pd.DataFrame,
    store: Store,
    regime,
    fired: list | None,
) -> None:
        today = session_date(dt.datetime.now())

        try:
            lists = portfolio.load_all()
            lists_error = False
        except Exception as e:
            log.warning("cannot read portfolios/watchlists: %s", e)
            lists = {}
            lists_error = True

        cache_key = _get_cache_key(store, today, fired)
        m = _cached_morning(cache_key, universe, store, today, lists, fired)

        if lists_error:
            hebrew_note(t("Could not load portfolios and watchlists."))
        if m.errors:
            hebrew_note(t("Some My Morning data is unavailable."))

        # 1. New in screens
        has_new_screens = False
        if isinstance(m.new_in_screens, pd.DataFrame) and not m.new_in_screens.empty:
            valid_new = m.new_in_screens[
                m.new_in_screens["new"].notna() & (m.new_in_screens["new"] > 0)
            ]
            if not valid_new.empty:
                has_new_screens = True
                st.markdown(f"#### {t('New in screens')}")
                rows_disp = []
                for _, r in valid_new.iterrows():
                    name = str(r.get("screen", ""))
                    if name.startswith("★ "):
                        disp_name = name
                    else:
                        disp_name = i18n.preset(name, L())
                    new_val = int(r["new"])
                    tot_raw = r.get("total", 0)
                    tot_val = int(tot_raw) if pd.notna(tot_raw) and tot_raw is not None else 0
                    rows_disp.append({
                        t("Screen"): disp_name,
                        t("New"): new_val,
                        t("Total"): tot_val,
                    })
                df_screens_disp = pd.DataFrame(rows_disp)
                st.dataframe(
                    df_screens_disp,
                    hide_index=True,
                    use_container_width=True,
                    key="hm_new_screens_table",
                )

        # 2. Movers
        has_movers = False
        if isinstance(m.movers, pd.DataFrame) and not m.movers.empty:
            valid_movers = m.movers[m.movers["change"].notna()].copy()
            if not valid_movers.empty:
                has_movers = True
                st.markdown(f"#### {t('Top movers in my lists')}")
                rows_disp = []
                for _, r in valid_movers.iterrows():
                    l_val = r.get("lists")
                    if isinstance(l_val, (list, tuple)):
                        l_str = ", ".join(str(x) for x in l_val if x)
                    else:
                        l_str = str(l_val) if pd.notna(l_val) else ""

                    rows_disp.append({
                        col_label("ticker"): r.get("ticker", "—"),
                        col_label("company"): r.get("company", "—"),
                        col_label("change"): float(r["change"]),
                        t("List"): l_str,
                    })
                df_movers_disp = pd.DataFrame(rows_disp)
                chg_col = col_label("change")
                styled_movers = df_movers_disp.style.map(_sign_color, subset=[chg_col]).format({
                    chg_col: "{:+.2f}%"
                })
                st.dataframe(
                    styled_movers,
                    hide_index=True,
                    use_container_width=True,
                    key="hm_movers_table",
                )

        # 3. Earnings
        has_earnings = False
        if isinstance(m.earnings, pd.DataFrame) and not m.earnings.empty:
            valid_earn = m.earnings[m.earnings["days"].notna()].copy()
            if not valid_earn.empty:
                has_earnings = True
                st.markdown(f"#### {t('Upcoming earnings')}")
                rows_disp = []
                for _, r in valid_earn.iterrows():
                    try:
                        days = int(r["days"])
                    except (ValueError, TypeError):
                        continue

                    if days == 0:
                        timing = "היום"
                    elif days == 1:
                        timing = "מחר"
                    else:
                        timing = f"בעוד {days} ימים"

                    edate = r.get("earnings_date")
                    if pd.notna(edate) and edate is not None:
                        if isinstance(edate, (dt.date, dt.datetime, pd.Timestamp)):
                            date_str = edate.strftime("%d/%m")
                        else:
                            try:
                                date_str = pd.to_datetime(edate).strftime("%d/%m")
                            except Exception:
                                date_str = str(edate)
                    else:
                        date_str = "—"

                    rows_disp.append({
                        col_label("ticker"): r.get("ticker", "—"),
                        col_label("company"): r.get("company", "—"),
                        col_label("earnings_date"): date_str,
                        t("When"): timing,
                    })
                df_earn_disp = pd.DataFrame(rows_disp)
                st.dataframe(
                    df_earn_disp,
                    hide_index=True,
                    use_container_width=True,
                    key="hm_earnings_table",
                )

        # 4. Alerts
        has_alerts = False
        alert_msgs = []
        alerts_source = fired if fired is not None else getattr(m, "alerts", [])
        if alerts_source:
            for item in alerts_source:
                msg = getattr(item, "message", None)
                if not msg and isinstance(item, str):
                    msg = item
                if msg:
                    alert_msgs.append(str(msg))

        if alert_msgs:
            has_alerts = True
            st.markdown(f"#### {t('Alerts')}")
            for msg in alert_msgs:
                st.info(msg, icon="🔔")

        # Empty state
        if not (has_new_screens or has_movers or has_earnings or has_alerts):
            st.info(t("Nothing new today"))

        # Daily report expander
        with st.expander(t("Daily report text"), expanded=False, key="hm_report_expander"):
            rep = daily_report.compose(m, regime, fired, today)
            st.code(rep.text, language=None)
            st.caption(t("Sending the report via Telegram will be added in a future update."))


render = home_page
