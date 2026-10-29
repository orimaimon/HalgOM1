"""ATR position sizer and stop-loss calculator UI panel component."""
from __future__ import annotations

import logging
import math

import pandas as pd
import streamlit as st

from screener.ui.common import col_label, hebrew_note, L, ss, t

log = logging.getLogger("screener.app")


def calculate_sizing(
    price: float,
    atr: float,
    account: float,
    risk_pct: float = 1.0,
    atr_mult: float = 2.0,
    target_r: float = 2.0,
) -> dict | None:
    """Calculate ATR-based position size, stop-loss price, target price and risk."""
    inputs = [price, atr, account, risk_pct, atr_mult, target_r]
    if any(v is None or pd.isna(v) or v <= 0 for v in inputs):
        return None

    if atr * atr_mult >= price:
        return None

    stop_dist = atr * atr_mult
    stop_price = price - stop_dist
    stop_pct = -(stop_dist / price * 100.0)
    risk_dollars = account * (risk_pct / 100.0)
    shares = min(int(math.floor(risk_dollars / stop_dist)), int(math.floor(account / price)))
    cost = shares * price
    weight_pct = cost / account * 100.0
    actual_risk = shares * stop_dist
    target_price = price + stop_dist * target_r
    target_pct = (stop_dist * target_r / price) * 100.0

    return {
        "stop_dist": stop_dist,
        "stop_price": stop_price,
        "stop_pct": stop_pct,
        "risk_dollars": risk_dollars,
        "shares": shares,
        "cost": cost,
        "weight_pct": weight_pct,
        "actual_risk": actual_risk,
        "target_price": target_price,
        "target_pct": target_pct,
    }


def render(row: pd.Series) -> None:
    """Render position sizer & stop calculator expander panel for a stock row."""
    try:
        with st.expander(t("Position sizer & stop calculator"), expanded=False):
            price = row.get("price")
            atr = row.get("atr")

            if price is None or pd.isna(price) or price <= 0 or atr is None or pd.isna(atr) or atr <= 0:
                hebrew_note(t("No ATR data for position sizing"))
                return

            c1, c2, c3 = st.columns(3)
            with c1:
                account = st.number_input(
                    t("Portfolio size ($)"), min_value=1000.0, value=50000.0, step=5000.0, key="sz_account"
                )
            with c2:
                risk_pct = st.number_input(
                    t("Risk per trade (%)"), min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="sz_risk_pct"
                )
            with c3:
                atr_mult = st.number_input(
                    t("Stop distance (ATR x)"), min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="sz_atr_mult"
                )

            plan = calculate_sizing(
                price=float(price),
                atr=float(atr),
                account=float(account),
                risk_pct=float(risk_pct),
                atr_mult=float(atr_mult),
            )

            if plan is None or plan["shares"] <= 0:
                hebrew_note(t("No ATR data for position sizing"))
                return

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    t("Recommended shares"),
                    f"{plan['shares']:,}",
                    delta=f"${plan['cost']:,.0f} ({plan['weight_pct']:.1f}%)",
                    delta_color="off",
                )
            with m2:
                st.metric(
                    t("Stop-loss price"),
                    f"${plan['stop_price']:,.2f}",
                    delta=f"{plan['stop_pct']:.1f}%",
                    delta_color="inverse",
                )
            with m3:
                st.metric(
                    t("Target price (2R)"),
                    f"${plan['target_price']:,.2f}",
                    delta=f"+{plan['target_pct']:.1f}%",
                    delta_color="normal",
                )
            with m4:
                st.metric(
                    t("Dollar risk"),
                    f"${plan['actual_risk']:,.0f}",
                )

            hebrew_note(
                f"קניית {plan['shares']:,} מניות לפי {plan['stop_pct']:.1f}% סטופ "
                f"תקצה ${plan['cost']:,.0f} ({plan['weight_pct']:.1f}% מהתיק) בסיכון מרבי של ${plan['actual_risk']:,.0f}."
            )
    except Exception:
        log.exception("Error rendering sizing panel")
        hebrew_note("שגיאה בחישוב גודל פוזיציה")


def render_sizing_panel(default_ticker: str = "", default_price: float = 0.0) -> None:
    """Interactive position sizing panel allowing input of account equity, risk %, entry price, and stop loss."""
    try:
        if default_ticker:
            st.subheader(f"מחשבון גודל פוזיציה — {default_ticker}")

        def_price = float(default_price) if default_price and default_price > 0 else 0.0
        def_stop = round(def_price * 0.95, 2) if def_price > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            account = st.number_input(
                "שווי תיק $",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                key=f"rsp_account_{default_ticker}",
            )
        with c2:
            risk_pct = st.number_input(
                "אחוז סיכון לטרייד %",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.25,
                key=f"rsp_risk_pct_{default_ticker}",
            )
        with c3:
            entry = st.number_input(
                "מחיר כניסה $",
                min_value=0.0,
                value=def_price,
                step=0.5,
                key=f"rsp_entry_{default_ticker}",
            )
        with c4:
            stop = st.number_input(
                "מחיר קטיעת הפסד Stop $",
                min_value=0.0,
                value=def_stop,
                step=0.5,
                key=f"rsp_stop_{default_ticker}",
            )

        if entry > 0 and stop > 0:
            if stop >= entry:
                st.warning("מחיר קטיעת הפסד Stop חייב להיות נמוך ממחיר כניסה")
                return

            stop_dist = entry - stop
            risk_dollars = account * (risk_pct / 100.0)
            if stop_dist > 0 and account > 0:
                shares_by_risk = int(math.floor(risk_dollars / stop_dist))
                shares_by_account = int(math.floor(account / entry))
                shares = max(0, min(shares_by_risk, shares_by_account))
            else:
                shares = 0

            position_value = shares * entry
            max_risk = shares * stop_dist
            weight_pct = (position_value / account * 100.0) if account > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("כמות מניות מומלצת", f"{shares:,}")
            with m2:
                st.metric("שווי פוזיציה $", f"${position_value:,.2f}")
            with m3:
                st.metric("סיכון דולרי כולל $", f"${max_risk:,.2f}")
            with m4:
                st.metric("משקל מהתיק %", f"{weight_pct:.1f}%")

            if weight_pct > 25.0:
                st.warning("משקל הפוזיציה חורג מ-25% משווי התיק")
    except Exception:
        log.exception("Error rendering interactive sizing panel")
        hebrew_note("שגיאה בחישוב גודל פוזיציה")
