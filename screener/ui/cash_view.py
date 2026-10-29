"""Cash & dividend management UI component.

Displays portfolio cash balance, total deposits, total dividends,
a table of cash transactions, and a form to add cash entries.

Public entry point:
render_cash_view(portfolio_name: str, cash_entries: list, on_add_entry=None) -> None
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import streamlit as st

from screener.ui.common import hebrew_note

log = logging.getLogger("screener.app")

DEPOSIT_KINDS = {"הפקדה", "deposit"}
WITHDRAWAL_KINDS = {"משיכה", "withdrawal"}
DIVIDEND_KINDS = {"דיבידנד", "dividend"}
INTEREST_KINDS = {"ריבית", "interest"}


def _get_field(entry: dict | object, key: str, default=None):
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _format_money(amount: float) -> str:
    if pd.isna(amount):
        return "$0.00"
    if amount < 0:
        return f"-${abs(amount):,.2f}"
    return f"${amount:,.2f}"


def render_cash_view(portfolio_name: str, cash_entries: list, on_add_entry=None) -> None:
    """Renders cash balance metrics, transaction history, and entry submission form."""
    try:
        entries = cash_entries or []

        total_deposits = 0.0
        total_withdrawals = 0.0
        total_dividends = 0.0
        total_interest = 0.0

        table_rows = []

        for entry in entries:
            kind_raw = str(_get_field(entry, "kind", "") or _get_field(entry, "type", "")).strip()
            kind_lower = kind_raw.lower()
            amount_raw = _get_field(entry, "amount", 0.0)
            try:
                amount = abs(float(amount_raw or 0.0))
            except (ValueError, TypeError):
                amount = 0.0

            if kind_raw in DEPOSIT_KINDS or kind_lower in DEPOSIT_KINDS:
                kind_display = "הפקדה"
                total_deposits += amount
            elif kind_raw in WITHDRAWAL_KINDS or kind_lower in WITHDRAWAL_KINDS:
                kind_display = "משיכה"
                total_withdrawals += amount
            elif kind_raw in DIVIDEND_KINDS or kind_lower in DIVIDEND_KINDS:
                kind_display = "דיבידנד"
                total_dividends += amount
            elif kind_raw in INTEREST_KINDS or kind_lower in INTEREST_KINDS:
                kind_display = "ריבית"
                total_interest += amount
            else:
                kind_display = kind_raw or "—"
                total_deposits += amount

            date_val = _get_field(entry, "date", "")
            if isinstance(date_val, (dt.date, dt.datetime, pd.Timestamp)):
                date_str = date_val.strftime("%d/%m/%Y")
            elif isinstance(date_val, str) and date_val:
                try:
                    date_parsed = dt.date.fromisoformat(date_val[:10])
                    date_str = date_parsed.strftime("%d/%m/%Y")
                except ValueError:
                    date_str = date_val
            else:
                date_str = "—"

            ticker_val = str(_get_field(entry, "ticker", "") or "").strip().upper()
            note_val = str(_get_field(entry, "note", "") or "").strip()

            table_rows.append({
                "תאריך": date_str,
                "סוג": kind_display,
                "סכום": _format_money(amount),
                "מניה": ticker_val if ticker_val else "—",
                "הערה": note_val if note_val else "—",
            })

        cash_balance = (total_deposits + total_dividends + total_interest) - total_withdrawals

        # a) Top summary metric cards
        c1, c2, c3 = st.columns(3)
        c1.metric("יתרת מזומן", _format_money(cash_balance))
        c2.metric("סך הפקדות", _format_money(total_deposits))
        c3.metric("דיבידנדים שהתקבלו", _format_money(total_dividends))

        # b) Table of recent cash transactions
        if table_rows:
            df = pd.DataFrame(table_rows)
        else:
            df = pd.DataFrame(columns=["תאריך", "סוג", "סכום", "מניה", "הערה"])

        st.dataframe(df, hide_index=True, use_container_width=True, key=f"cs_table_{portfolio_name}")

        # c) Expander form "הוספת תנועת מזומן"
        with st.expander("הוספת תנועת מזומן"):
            with st.form(key=f"cs_form_{portfolio_name}"):
                date_val = st.date_input("תאריך", value=dt.date.today(), key=f"cs_date_{portfolio_name}")
                kind_val = st.selectbox(
                    "סוג",
                    options=["הפקדה", "משיכה", "דיבידנד", "ריבית"],
                    key=f"cs_kind_{portfolio_name}",
                )
                amount_val = st.number_input(
                    "סכום ($)",
                    min_value=0.0,
                    step=10.0,
                    key=f"cs_amount_{portfolio_name}",
                )
                ticker_val = st.text_input("מניה (אופציונלי)", key=f"cs_ticker_{portfolio_name}")
                note_val = st.text_input("הערה", key=f"cs_note_{portfolio_name}")

                submitted = st.form_submit_button("הוספת תנועה")
                if submitted:
                    if on_add_entry is not None:
                        new_entry = {
                            "date": date_val,
                            "kind": kind_val,
                            "amount": float(amount_val),
                            "ticker": ticker_val.strip().upper(),
                            "note": note_val.strip(),
                        }
                        on_add_entry(portfolio_name, new_entry)

    except Exception:
        log.exception("error rendering cash view for %s", portfolio_name)
        hebrew_note("שגיאה בהצגת נתוני מזומנים")
