"""Israeli capital gains tax report UI component.

Renders annual capital gains tax estimates for closed portfolio trades,
summarizing proceeds, gains, losses, and estimated 25% tax liability.

Public entry point: render_tax_report(tax_df: pd.DataFrame, summary: dict, year: int) -> None
"""
from __future__ import annotations

import datetime as dt
import logging
import pandas as pd
import streamlit as st

from screener.ui.common import hebrew_note, t

log = logging.getLogger("screener.app")


def _fmt_ils(v) -> str:
    """Format float or int as ILS currency string."""
    if pd.isna(v) or v is None:
        return "₪0.00"
    try:
        val = float(v)
        return f"₪{val:,.2f}"
    except (ValueError, TypeError):
        return str(v)


def _fmt_rate(v) -> str:
    """Format exchange rate."""
    if pd.isna(v) or v is None:
        return "—"
    try:
        val = float(v)
        return f"{val:.4f}"
    except (ValueError, TypeError):
        return str(v)


def _fmt_date(v) -> str:
    """Format date value as DD/MM/YYYY or YYYY-MM-DD string."""
    if pd.isna(v) or v is None:
        return "—"
    if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
        return v.strftime("%d/%m/%Y")
    s = str(v).strip()
    if len(s) >= 10 and s[:10].replace("-", "").isdigit():
        try:
            d = dt.date.fromisoformat(s[:10])
            return d.strftime("%d/%m/%Y")
        except ValueError:
            pass
    return s if s else "—"


def render_tax_report(tax_df: pd.DataFrame, summary: dict, year: int) -> None:
    """Render Israeli capital gains tax estimate report.

    Args:
        tax_df: DataFrame of closed trades for tax report.
        summary: Summary metrics dictionary for the target year.
        year: Target tax year integer.
    """
    try:
        # Determine available years for selector
        years_set = {int(year)}
        if tax_df is not None and not tax_df.empty:
            for col in ["sell_date", "date", "תאריך מכירה"]:
                if col in tax_df.columns:
                    for d_val in tax_df[col].dropna():
                        if isinstance(d_val, (dt.date, dt.datetime, pd.Timestamp)):
                            years_set.add(d_val.year)
                        elif isinstance(d_val, str) and len(d_val) >= 4 and d_val[:4].isdigit():
                            try:
                                years_set.add(int(d_val[:4]))
                            except ValueError:
                                pass

        available_years = sorted(list(years_set), reverse=True)
        default_idx = available_years.index(int(year)) if int(year) in available_years else 0

        # a) Year selector / filter
        selected_year = st.selectbox(
            "שנת מס",
            options=available_years,
            index=default_idx,
            key="tx_year",
        )

        # Filter tax_df by selected year if date column exists
        filtered_df = tax_df.copy() if tax_df is not None else pd.DataFrame()
        if not filtered_df.empty:
            date_col = None
            for c in ["sell_date", "date", "תאריך מכירה"]:
                if c in filtered_df.columns:
                    date_col = c
                    break
            if date_col:
                def _get_yr(val):
                    if isinstance(val, (dt.date, dt.datetime, pd.Timestamp)):
                        return val.year
                    if isinstance(val, str) and len(val) >= 4 and val[:4].isdigit():
                        try:
                            return int(val[:4])
                        except ValueError:
                            return None
                    return None

                yr_series = filtered_df[date_col].map(_get_yr)
                if yr_series.notna().any():
                    filtered_df = filtered_df[yr_series == selected_year]

        # Extract summary metrics with safe defaults
        s_dict = summary or {}
        proceeds = s_dict.get("total_proceeds_ils", s_dict.get("proceeds_ils", s_dict.get("proceeds", 0.0)))
        gain = s_dict.get("net_gain_ils", s_dict.get("total_gain_ils", s_dict.get("gain_ils", s_dict.get("gain", 0.0))))
        losses = s_dict.get("total_loss_ils", s_dict.get("loss_ils", s_dict.get("losses", 0.0)))
        tax = s_dict.get("estimated_tax_ils", s_dict.get("tax_ils", s_dict.get("tax", 0.0)))

        # Fallback metric calculations from filtered_df if summary keys missing
        if proceeds == 0.0 and not filtered_df.empty:
            for p_col in ["proceeds_ils", "total_proceeds_ils"]:
                if p_col in filtered_df.columns:
                    proceeds = float(filtered_df[p_col].sum())
                    break

        # b) Top summary metric cards using st.metric
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("סך תמורה בש\"ח", _fmt_ils(proceeds))
        c2.metric("רווח הון ריאלי/נומינלי בש\"ח", _fmt_ils(gain))
        c3.metric("הפסדים לקיזוז", _fmt_ils(losses))
        c4.metric("חבות מס משוערת 25%", _fmt_ils(tax))

        # c) Table of closed trades
        # Target exact columns: נייר, תאריך קנייה, שער דולר קנייה, תאריך מכירה, שער דולר מכירה, רווח/הפסד בש"ח, מס משוער
        target_cols = [
            "נייר",
            "תאריך קנייה",
            "שער דולר קנייה",
            "תאריך מכירה",
            "שער דולר מכירה",
            "רווח/הפסד בש\"ח",
            "מס משוער",
        ]

        if filtered_df.empty:
            display_df = pd.DataFrame(columns=target_cols)
        else:
            rows = []
            for _, r in filtered_df.iterrows():
                tkr = r.get("ticker", r.get("נייר", "—"))
                b_date = r.get("buy_date", r.get("תאריך קנייה", "—"))
                b_rate = r.get("buy_rate", r.get("usd_ils_buy", r.get("buy_usd_ils", r.get("שער דולר קנייה", "—"))))
                s_date = r.get("sell_date", r.get("date", r.get("תאריך מכירה", "—")))
                s_rate = r.get("sell_rate", r.get("usd_ils_sell", r.get("sell_usd_ils", r.get("שער דולר מכירה", "—"))))
                pnl = r.get("nominal_gain_ils", r.get("gain_ils", r.get("pnl_ils", r.get("רווח/הפסד בש\"ח", 0.0))))
                est_tax = r.get("estimated_tax_ils", r.get("tax_ils", r.get("מס משוער", 0.0)))

                rows.append({
                    "נייר": str(tkr),
                    "תאריך קנייה": _fmt_date(b_date),
                    "שער דולר קנייה": _fmt_rate(b_rate),
                    "תאריך מכירה": _fmt_date(s_date),
                    "שער דולר מכירה": _fmt_rate(s_rate),
                    "רווח/הפסד בש\"ח": _fmt_ils(pnl),
                    "מס משוער": _fmt_ils(est_tax),
                })
            display_df = pd.DataFrame(rows, columns=target_cols)

        st.dataframe(display_df, hide_index=True, use_container_width=True, key="tx_trades_table")

        # d) Legal disclaimer callout in Hebrew
        st.info("החישוב נועד לאומדן כללי בלבד ואינו מהווה ייעוץ מס או תחליף לדוח רשמי.")

    except Exception:
        log.exception("error rendering tax report")
        hebrew_note("שגיאה בהצגת דוח מס")
