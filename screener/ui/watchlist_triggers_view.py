"""Smart Watchlist Triggers UI Component.

Displays target entry prices, trader notes, triggered stock alerts,
portfolio overlap, and form controls to update targets and notes.
"""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from screener import portfolio, watchlist_triggers
from screener.portfolio import WATCHLIST, PList
from screener.ui.common import L, col_label, hebrew_note, ss, t

log = logging.getLogger("screener.app")


def _render_body(plist: PList, universe: pd.DataFrame) -> None:
    if plist.kind != WATCHLIST or not plist.rows:
        return

    all_lists = portfolio.load_all()
    df = watchlist_triggers.evaluate_watchlist(plist, universe, all_lists=all_lists)

    # 1. Banners for triggered stocks
    if not df.empty and "status" in df.columns:
        triggered_rows = df[df["status"] == "triggered"]
        for _, r in triggered_rows.iterrows():
            ticker = r["ticker"]
            curr = r["current"]
            target = r["target_price"]
            curr_str = f"${curr:.2f}" if pd.notna(curr) else "—"
            target_str = f"${target:.2f}" if pd.notna(target) else "—"
            st.success(f"🎯 {ticker} הגיעה למחיר היעד ({curr_str} מול יעד {target_str})")

    # 2. Enriched table
    if not df.empty:
        st.write(f"### {t('Trading playbook & triggers')}")
        disp_df = df.copy()

        # Format columns for display
        if "direction" in disp_df.columns:
            disp_df["direction"] = disp_df["direction"].map(lambda d: t(d) if pd.notna(d) else "—")
        if "in_portfolio" in disp_df.columns:
            disp_df["in_portfolio"] = disp_df["in_portfolio"].map(lambda p: "✓" if p else "—")

        st.dataframe(disp_df, hide_index=True, use_container_width=True, key="wt_table")

    st.write("")

    # 3. Form/controls to edit target price, direction, and note
    tickers = [str(r.get("ticker", "")).strip() for r in plist.rows if r.get("ticker")]
    if not tickers:
        return

    selected_ticker = st.selectbox(t("Ticker"), options=tickers, key="wt_ticker")

    # Load existing values for selected_ticker
    targets = watchlist_triggers.load_targets()
    target_key = f"{plist.name}:{selected_ticker}"
    target_obj = targets.get(target_key)

    existing_target = target_obj.target_price if target_obj and target_obj.target_price is not None else 0.0
    existing_dir = target_obj.direction if target_obj and target_obj.direction in ("above", "below") else "above"

    existing_note = ""
    for r in plist.rows:
        if r.get("ticker") == selected_ticker:
            existing_note = str(r.get("note", "")) if r.get("note") and pd.notna(r.get("note")) else ""
            break

    dir_options = ["above", "below"]
    dir_index = dir_options.index(existing_dir) if existing_dir in dir_options else 0

    c1, c2, c3 = st.columns([2, 2, 3])
    target_val = c1.number_input(
        t("Target entry price"),
        min_value=0.0,
        step=0.5,
        value=float(existing_target),
        key="wt_target",
    )
    direction_val = c2.radio(
        t("Trigger direction"),
        options=dir_options,
        index=dir_index,
        format_func=t,
        key="wt_dir",
    )
    note_val = c3.text_input(
        t("Notes"),
        value=existing_note,
        key="wt_note",
    )

    if st.button(t("Save plan"), key="wt_save"):
        target_to_save = float(target_val) if target_val > 0 else None
        watchlist_triggers.set_target(
            list_name=plist.name,
            ticker=selected_ticker,
            target_price=target_to_save,
            direction=direction_val,
        )
        watchlist_triggers.update_note(
            plist=plist,
            ticker=selected_ticker,
            note=note_val.strip(),
        )
        st.rerun()


def render(plist: PList, universe: pd.DataFrame) -> None:
    """Render the Watchlist Triggers UI component."""
    if plist.kind != WATCHLIST or not plist.rows:
        return

    try:
        _render_body(plist, universe)
    except Exception:
        log.exception("Error rendering watchlist triggers")
        hebrew_note(t("Error displaying watchlist triggers"))
