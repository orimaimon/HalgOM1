"""
Morning Dashboard component ("הבוקר שלי") displaying daily market highlights,
portfolio moves, upcoming earnings, and new scan candidates.
"""
from __future__ import annotations

import datetime as dt
import logging
import pandas as pd
import streamlit as st

from screener import i18n, theme
from screener.portfolio import PORTFOLIO, PList, positions
from screener.ui.common import L, col_label, hebrew_note, t

log = logging.getLogger("screener.app")


def render_morning_dashboard(
    universe_df: pd.DataFrame,
    market_df: pd.DataFrame = None,
    portfolios_data: dict = None,
) -> None:
    """Renders the Morning Dashboard summary cards."""
    try:
        _render_dashboard(universe_df, market_df, portfolios_data)
    except Exception as e:
        log.exception("Error rendering morning dashboard: %s", e)
        hebrew_note(t("Could not load My Morning data right now."))


def _sign_color(val):
    if isinstance(val, (int, float)) and pd.notna(val) and val != 0:
        return f"color: {theme.GAIN}" if val > 0 else f"color: {theme.LOSS}"
    return ""


def _extract_list_frame(plist_obj) -> pd.DataFrame:
    if isinstance(plist_obj, PList):
        return plist_obj.frame()
    if isinstance(plist_obj, pd.DataFrame):
        return plist_obj
    if isinstance(plist_obj, dict) and "rows" in plist_obj:
        return pd.DataFrame(plist_obj["rows"])
    return pd.DataFrame()


def _get_list_kind(plist_obj) -> str:
    if isinstance(plist_obj, PList):
        return getattr(plist_obj, "kind", PORTFOLIO)
    if isinstance(plist_obj, dict):
        return plist_obj.get("kind", PORTFOLIO)
    return PORTFOLIO


def _render_dashboard(
    universe_df: pd.DataFrame,
    market_df: pd.DataFrame = None,
    portfolios_data: dict = None,
) -> None:
    if universe_df is None:
        universe_df = pd.DataFrame()
    if portfolios_data is None:
        portfolios_data = {}

    st.subheader(t("הבוקר שלי"))

    # 1. Market Overview Banner (SPY, QQQ, VIX)
    st.markdown(f"#### {t('סקירת שוק')}")
    col1, col2, col3 = st.columns(3)

    market_rendered = False
    if market_df is not None and isinstance(market_df, pd.DataFrame) and not market_df.empty:
        # Expected indices: SPY, QQQ, VIX
        symbols_map = [
            ("SPY", col1, "SPY"),
            ("QQQ", col2, "QQQ"),
            ("VIX", col3, "VIX"),
        ]
        for sym, col_widget, label in symbols_map:
            if sym in market_df.index:
                row = market_df.loc[sym]
                px = row.get("close")
                chg = row.get("change")

                px_str = f"${float(px):,.2f}" if pd.notna(px) else "—"
                chg_str = f"{float(chg):+.2f}%" if pd.notna(chg) else None

                with col_widget:
                    st.metric(label=label, value=px_str, delta=chg_str)
                market_rendered = True
            else:
                with col_widget:
                    st.metric(label=label, value="—")

    if not market_rendered:
        st.info(t("אין נתוני שוק זמינים כעת"))

    # 2. Portfolio Movers
    st.markdown(f"#### {t('Top movers in my lists')}")
    ticker_to_lists: dict[str, list[str]] = {}
    for list_name, plist_obj in portfolios_data.items():
        frame = _extract_list_frame(plist_obj)
        if not frame.empty and "ticker" in frame.columns:
            for ticker in frame["ticker"].dropna().astype(str).unique():
                ticker_to_lists.setdefault(ticker, []).append(str(list_name))

    has_movers = False
    if ticker_to_lists and not universe_df.empty and "ticker" in universe_df.columns:
        univ_indexed = universe_df.set_index("ticker")
        candidates = []
        for t_sym, list_names in ticker_to_lists.items():
            if t_sym in univ_indexed.index:
                row = univ_indexed.loc[t_sym]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                chg = row.get("change")
                if pd.notna(chg):
                    candidates.append({
                        "ticker": t_sym,
                        "company": row.get("company", "—"),
                        "price": float(row.get("price")) if pd.notna(row.get("price")) else 0.0,
                        "change": float(chg),
                        "lists": ", ".join(list_names),
                        "_abs_change": abs(float(chg)),
                    })

        if candidates:
            candidates.sort(key=lambda x: x["_abs_change"], reverse=True)
            top_movers = candidates[:5]
            for c in top_movers:
                c.pop("_abs_change", None)

            movers_df = pd.DataFrame(top_movers)
            display_df = pd.DataFrame({
                col_label("ticker"): movers_df["ticker"],
                col_label("company"): movers_df["company"],
                col_label("price"): movers_df["price"].map(lambda p: f"${p:,.2f}" if p else "—"),
                col_label("change"): movers_df["change"],
                t("List"): movers_df["lists"],
            })

            chg_col = col_label("change")
            styled_movers = display_df.style.map(_sign_color, subset=[chg_col]).format({
                chg_col: "{:+.2f}%"
            })
            st.dataframe(
                styled_movers,
                hide_index=True,
                use_container_width=True,
                key="mv_movers_table",
            )
            has_movers = True

    if not has_movers:
        st.info(t("אין תנועות בולטות בתיקים להיום"))

    # 3. Upcoming Earnings (<= 14 days)
    st.markdown(f"#### {t('Upcoming earnings')}")
    held_tickers = set()
    for plist_obj in portfolios_data.values():
        kind = _get_list_kind(plist_obj)
        frame = _extract_list_frame(plist_obj)
        if not frame.empty:
            if kind == PORTFOLIO:
                pos = positions(frame)
                if not pos.empty and "quantity" in pos.columns:
                    held = pos[pos["quantity"] > 0]["ticker"].dropna().astype(str).unique()
                    held_tickers.update(held)
            else:
                if "ticker" in frame.columns:
                    held_tickers.update(frame["ticker"].dropna().astype(str).unique())

    has_earnings = False
    if held_tickers and not universe_df.empty and "ticker" in universe_df.columns:
        univ_indexed = universe_df.set_index("ticker")
        today_date = dt.date.today()
        earn_rows = []

        for t_sym in held_tickers:
            if t_sym in univ_indexed.index:
                row = univ_indexed.loc[t_sym]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                days_val = row.get("days_to_earnings")
                edate = row.get("earnings_date")

                if pd.notna(days_val):
                    try:
                        days_diff = int(days_val)
                    except (ValueError, TypeError):
                        days_diff = None
                elif pd.notna(edate):
                    ed_ts = pd.to_datetime(edate, errors="coerce")
                    if pd.notna(ed_ts):
                        days_diff = (ed_ts.date() - today_date).days
                    else:
                        days_diff = None
                else:
                    days_diff = None

                if days_diff is not None and 0 <= days_diff <= 14:
                    if pd.notna(edate):
                        ed_ts = pd.to_datetime(edate, errors="coerce")
                        date_str = ed_ts.strftime("%d/%m") if pd.notna(ed_ts) else str(edate)
                    else:
                        date_str = "—"

                    if days_diff == 0:
                        timing = t("היום")
                    elif days_diff == 1:
                        timing = t("מחר")
                    else:
                        timing = f"{t('בעוד')} {days_diff} {t('ימים')}"

                    earn_rows.append({
                        col_label("ticker"): t_sym,
                        col_label("company"): row.get("company", "—"),
                        col_label("earnings_date"): date_str,
                        t("When"): timing,
                        "_days": days_diff,
                    })

        if earn_rows:
            earn_rows.sort(key=lambda x: (x["_days"], x[col_label("ticker")]))
            for r in earn_rows:
                r.pop("_days", None)

            earn_df = pd.DataFrame(earn_rows)
            st.dataframe(
                earn_df,
                hide_index=True,
                use_container_width=True,
                key="mv_earnings_table",
            )
            has_earnings = True

    if not has_earnings:
        st.info(t("אין דוחות כספיים קרובים ב-14 הימים הקרובים"))

    # 4. New Scan Candidates
    st.markdown(f"#### {t('מועמדים חדשים בסריקות')}")
    new_candidates_count = 0
    if not universe_df.empty:
        if "new" in universe_df.columns:
            new_col = universe_df["new"]
            if new_col.dtype == bool:
                new_candidates_count = int(new_col.sum())
            else:
                new_candidates_count = int(new_col.isin([True, 1]).sum())
        elif "is_new" in universe_df.columns:
            new_col = universe_df["is_new"]
            if new_col.dtype == bool:
                new_candidates_count = int(new_col.sum())
            else:
                new_candidates_count = int(new_col.isin([True, 1]).sum())

    st.metric(label=t("מספר מניות חדשות בסריקות היום"), value=str(new_candidates_count))
