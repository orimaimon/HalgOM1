"""Shekel P&L UI component for portfolios.

Renders open position metrics in shekels (ILS) with stock vs currency P&L split,
plus realized sales in ILS.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import streamlit as st

from screener import portfolio_ils, theme
from screener.data import boi, guard, store
from screener.portfolio import PORTFOLIO, PList
from screener.ui.common import col_label, hebrew_note, L, ss, t

log = logging.getLogger("screener.app")


def _fmt_ils(v) -> str:
    if pd.isna(v):
        return "—"
    return f"₪{round(v):,}"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def _color_pnl(v) -> str:
    if isinstance(v, (int, float)) and not pd.isna(v) and v != 0:
        return f"color: {theme.GAIN}" if v > 0 else f"color: {theme.LOSS}"
    return ""


def render(plist: PList, universe: pd.DataFrame) -> None:
    """Render shekel P&L view for open positions and realized trades."""
    if plist is None or getattr(plist, "kind", None) != PORTFOLIO:
        return

    try:
        if st.button(t("Update USD/ILS rates from Bank of Israel"), key="ils_update"):
            trades = plist.frame()
            if not trades.empty and "date" in trades.columns and trades["date"].notna().any():
                earliest = trades["date"].dropna().min()
                if isinstance(earliest, (dt.datetime, pd.Timestamp)):
                    earliest = earliest.date()
            else:
                earliest = dt.date.today() - dt.timedelta(days=365)

            updated = boi.update(start=earliest)
            if updated:
                st.toast(t("Updated"), icon="✅")
            else:
                g_status = guard.status("boi")
                if g_status.get("wait_until"):
                    try:
                        until_dt = dt.datetime.fromisoformat(g_status["wait_until"])
                        w = guard.Wait("boi", until_dt, g_status.get("wait_reason") or "backoff")
                        st.info(guard.wait_message(w))
                    except Exception:
                        st.info(t("No new rates"))
                else:
                    st.info(t("No new rates"))

        rates = boi.load_rates()
        if rates.empty:
            st.info(t("No saved rates yet — click update"))

        prices = (
            universe.set_index("ticker")["price"]
            if isinstance(universe, pd.DataFrame) and "ticker" in universe.columns and "price" in universe.columns
            else pd.Series(dtype=float)
        )

        today = store.market_date(dt.datetime.now().astimezone())
        df_pos = portfolio_ils.ils_positions(plist, prices, rates, today)

        if not df_pos.empty:
            disp_cols = ["ticker", "cost_ils", "value_ils", "pnl_ils", "pnl_ils_pct", "stock_part_ils", "fx_part_ils"]
            disp_df = df_pos[disp_cols].copy()
            col_map = {
                "ticker": t("Ticker"),
                "cost_ils": t("Cost ₪"),
                "value_ils": t("Value ₪"),
                "pnl_ils": t("P&L ₪"),
                "pnl_ils_pct": t("P&L %"),
                "stock_part_ils": t("Stock part ₪"),
                "fx_part_ils": t("Currency part ₪"),
            }
            disp_df.rename(columns=col_map, inplace=True)

            c_pnl = col_map["pnl_ils"]
            c_pct = col_map["pnl_ils_pct"]
            c_stock = col_map["stock_part_ils"]
            c_fx = col_map["fx_part_ils"]

            fmt_dict = {
                col_map["cost_ils"]: _fmt_ils,
                col_map["value_ils"]: _fmt_ils,
                c_pnl: _fmt_ils,
                c_pct: _fmt_pct,
                c_stock: _fmt_ils,
                c_fx: _fmt_ils,
            }

            st.dataframe(
                disp_df.style.map(_color_pnl, subset=[c_pnl, c_pct, c_stock, c_fx]).format(fmt_dict),
                hide_index=True,
                use_container_width=True,
                key="ils_positions_table",
            )

            tot_pnl = df_pos["pnl_ils"].sum(min_count=1)
            tot_stock = df_pos["stock_part_ils"].sum(min_count=1)
            tot_fx = df_pos["fx_part_ils"].sum(min_count=1)

            pnl_str = _fmt_ils(tot_pnl) if pd.notna(tot_pnl) else "—"
            stock_str = _fmt_ils(tot_stock) if pd.notna(tot_stock) else "—"
            fx_str = _fmt_ils(tot_fx) if pd.notna(tot_fx) else "—"

            hebrew_note(f"**{t('Total: P&L')}** {pnl_str} · **{t('of which stock')}** {stock_str} · **{t('of which FX')}** {fx_str}")

            no_rate = df_pos.attrs.get("no_rate", [])
            if no_rate:
                hebrew_note(f"⚠️ {t('No exchange rates for:')} {', '.join(no_rate)}")

        hebrew_note("חלק המניה מחושב לפי שינוי מחיר המניה בשער הקנייה הממוצע; חלק השער מחושב לפי שינוי שער הדולר על השווי הנוכחי.")

        with st.expander(t("Realized P&L (ILS)")):
            hebrew_note(t("For assistance only — not a tax calculation"))
            df_realized = portfolio_ils.realized_ils(plist, rates)
            if not df_realized.empty:
                r_cols = ["date", "ticker", "quantity", "proceeds_ils", "cost_ils", "pnl_ils"]
                r_df = df_realized[r_cols].copy()
                r_col_map = {
                    "date": t("Date"),
                    "ticker": t("Ticker"),
                    "quantity": t("Quantity"),
                    "proceeds_ils": t("Proceeds ₪"),
                    "cost_ils": t("Cost ₪"),
                    "pnl_ils": t("P&L ₪"),
                }
                r_df.rename(columns=r_col_map, inplace=True)
                r_pnl_col = r_col_map["pnl_ils"]
                r_fmt = {
                    r_col_map["proceeds_ils"]: _fmt_ils,
                    r_col_map["cost_ils"]: _fmt_ils,
                    r_pnl_col: _fmt_ils,
                }
                st.dataframe(
                    r_df.style.map(_color_pnl, subset=[r_pnl_col]).format(r_fmt),
                    hide_index=True,
                    use_container_width=True,
                    key="ils_realized_table",
                )

    except Exception as e:
        log.exception("Failed to render Shekel P&L view: %s", e)
        hebrew_note(t("Failed to calculate shekel P&L"))
