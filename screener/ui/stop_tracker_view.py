"""Portfolio trailing-stop and target tracker UI component.

Renders ATR-based trailing stops, stop status warnings, open positions table, and quick alert creation.

Public entry point: render(plist: PList, universe: pd.DataFrame) -> None
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import streamlit as st

from screener import alerts, portfolio
from screener.portfolio import WATCHLIST, PList
from screener.ui.common import col_label, hebrew_note, L, ss, t

log = logging.getLogger("screener.app")


def _status(price: float, stop_price: float, distance_pct: float) -> str:
    if pd.isna(price) or pd.isna(stop_price) or price <= 0 or stop_price <= 0:
        return "אין נתון"
    if price <= stop_price:
        return "חצה סטופ"
    if 0 < distance_pct <= 1.5:
        return "קרוב לסטופ"
    if distance_pct > 1.5:
        return "תקין"
    return "אין נתון"


def render(plist: PList, universe: pd.DataFrame) -> None:
    """Render trailing stops & targets tracker for open portfolio positions."""
    try:
        if plist is None or getattr(plist, "kind", None) == WATCHLIST:
            hebrew_note(t("No open positions"))
            return

        trades = plist.frame()
        if trades.empty:
            hebrew_note(t("No open positions"))
            return

        pos = portfolio.positions(trades)
        open_pos = pos[pos["quantity"] > 0].copy() if not pos.empty and "quantity" in pos.columns else pd.DataFrame()
        if open_pos.empty:
            hebrew_note(t("No open positions"))
            return

        # Prepare lookup dicts from universe
        u_dict = {}
        if isinstance(universe, pd.DataFrame) and not universe.empty and "ticker" in universe.columns:
            u_sub = universe.drop_duplicates("ticker").set_index("ticker")
            u_dict = u_sub.to_dict("index")

        atr_mult = st.number_input(
            label="ATR multiplier",
            value=2.0,
            min_value=0.5,
            max_value=5.0,
            step=0.5,
            key="st_atr_mult",
        )

        rows = []
        for _, r in open_pos.iterrows():
            ticker = r["ticker"]
            avg_cost = r.get("avg_cost", np.nan)

            u_row = u_dict.get(ticker, {})
            company = u_row.get("company", "—")
            price = u_row.get("price", np.nan)
            atr = u_row.get("atr", np.nan)
            high_1m = u_row.get("high_1m", np.nan)

            # High ref = max of valid values among (avg_cost, price, high_1m)
            valid_refs = [v for v in (avg_cost, price, high_1m) if pd.notna(v) and v is not None]
            ref_high = max(valid_refs) if valid_refs else np.nan

            if pd.notna(price) and price > 0 and pd.notna(atr) and atr > 0 and pd.notna(ref_high):
                stop_price = ref_high - atr_mult * atr
                distance_pct = (price - stop_price) / price * 100.0
                target_price = avg_cost + 2.0 * atr_mult * atr if pd.notna(avg_cost) else np.nan
            else:
                stop_price = np.nan
                distance_pct = np.nan
                target_price = np.nan

            st_val = _status(price, stop_price, distance_pct)

            rows.append({
                "ticker": ticker,
                "company": company,
                "avg_cost": avg_cost,
                "price": price,
                "stop_price": stop_price,
                "distance_pct": distance_pct,
                "target_price": target_price,
                "status": st_val,
            })

        df_res = pd.DataFrame(rows)

        # 1. Show warnings
        for _, r in df_res.iterrows():
            st_val = r["status"]
            tk = r["ticker"]
            px = r["price"]
            sp = r["stop_price"]
            dist = r["distance_pct"]
            if st_val == "חצה סטופ":
                st.warning(f"⚠️ {tk}: המחיר (${px:.2f}) חצה את הסטופ הנגרר (${sp:.2f})")
            elif st_val == "קרוב לסטופ":
                st.warning(f"⚠️ {tk}: מרחק של {dist:.1f}% בלבד מהסטופ הנגרר (${sp:.2f})")

        # 2. Display Table
        disp_df = pd.DataFrame()
        disp_df[col_label("ticker")] = df_res["ticker"]
        disp_df[col_label("company")] = df_res["company"]
        disp_df[t("Avg cost")] = df_res["avg_cost"].map(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
        disp_df[t("Price")] = df_res["price"].map(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
        disp_df[t("Trailing stop")] = df_res["stop_price"].map(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
        disp_df[t("Distance to stop")] = df_res["distance_pct"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        disp_df[t("Target (2R)")] = df_res["target_price"].map(lambda v: f"${v:.2f}" if pd.notna(v) else "—")
        disp_df[t("Stop status")] = df_res["status"]

        st.dataframe(disp_df, hide_index=True, use_container_width=True, key="st_table")

        # 3. Quick alert creation bar
        alert_candidates = df_res[df_res["stop_price"].notna()].copy()
        if not alert_candidates.empty:
            open_tickers = alert_candidates["ticker"].tolist()
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            sel_ticker = c1.selectbox(t("Ticker"), options=open_tickers, key="st_ticker")
            if c2.button(t("Create stop alert"), key="st_set_alert"):
                target_row = alert_candidates[alert_candidates["ticker"] == sel_ticker].iloc[0]
                sp_val = target_row["stop_price"]
                alert_obj = alerts.Alert(
                    ticker=sel_ticker,
                    kind="price_below",
                    value=round(sp_val, 2),
                    note="סטופ נגרר ATR",
                )
                alerts.add(alert_obj)
                st.toast(t("Alert saved"))

    except Exception:
        log.exception("error rendering stop tracker view")
        hebrew_note(t("Error displaying stop tracker"))
