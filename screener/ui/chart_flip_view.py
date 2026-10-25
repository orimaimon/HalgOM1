"""Fast chart review (chart flip) UI component."""
from __future__ import annotations

import datetime as dt
import logging
import pandas as pd
import streamlit as st

from screener import explain, i18n, portfolio
from screener.theme import GAIN, LOSS
from screener.ui import explain_panel
from screener.ui.common import L, col_label, hebrew_note, ss, t
from screener.ui.screener_page import tradingview_chart

log = logging.getLogger("screener.app")


def _escape_md(text) -> str:
    if text is None or pd.isna(text):
        return ""
    s = str(text)
    if s.lower() in ("nan", "none", "—"):
        return ""
    for ch in ("\\", "*", "_", "`", "~", "$", "[", "]"):
        s = s.replace(ch, "\\" + ch)
    return s


def render(res: pd.DataFrame, selections: dict[str, str] | None = None) -> None:
    """Render fast chart review panel for stepping through screener results."""
    try:
        if res is None or res.empty or "ticker" not in res.columns:
            hebrew_note(t("No stocks to show"))
            return

        tickers = [str(x) for x in res["ticker"].tolist()]
        n = len(tickers)

        if "cf_idx" not in ss or not isinstance(ss["cf_idx"], int):
            ss["cf_idx"] = 0

        # Clamp index to [0, n - 1]
        idx = max(0, min(ss["cf_idx"], n - 1))
        ss["cf_idx"] = idx

        # Navigation bar
        c1, c2, c3, c4 = st.columns([1, 1, 2, 3], vertical_alignment="center")

        with c1:
            if st.button(t("Previous stock"), key="cf_prev", use_container_width=True):
                idx = (idx - 1) % n
                ss["cf_idx"] = idx
                ss["cf_jump"] = tickers[idx]
                st.rerun()

        with c2:
            if st.button(t("Next stock"), key="cf_next", use_container_width=True):
                idx = (idx + 1) % n
                ss["cf_idx"] = idx
                ss["cf_jump"] = tickers[idx]
                st.rerun()

        # Handle selectbox jump if changed
        if "cf_jump" in ss and ss["cf_jump"] in tickers:
            jump_idx = tickers.index(ss["cf_jump"])
            if jump_idx != idx:
                idx = jump_idx
                ss["cf_idx"] = idx

        with c3:
            ss["cf_jump"] = tickers[idx]
            st.selectbox(
                t("Jump to stock"),
                options=tickers,
                index=idx,
                key="cf_jump",
                label_visibility="collapsed",
            )

        row = res.iloc[idx]

        with c4:
            company = _escape_md(row.get("company"))
            comp_str = f" {company}" if company else ""
            st.markdown(f"**{idx + 1} / {n}** · **{row['ticker']}**{comp_str}")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)

        price_val = row.get("price")
        p_str = f"${price_val:,.2f}" if (price_val is not None and not pd.isna(price_val)) else "—"
        change_val = row.get("change")
        ch_str = f"{change_val:+.2f}%" if (change_val is not None and not pd.isna(change_val)) else None
        m1.metric(col_label("price"), p_str, ch_str)

        score_val = row.get("score")
        sc_str = f"{score_val:.0f}" if (score_val is not None and not pd.isna(score_val)) else "—"
        m2.metric(col_label("score"), sc_str)

        atr_val = row.get("atr_pct")
        atr_str = f"{atr_val:.2f}%" if (atr_val is not None and not pd.isna(atr_val)) else "—"
        m3.metric(col_label("atr_pct"), atr_str)

        sig_val = row.get("signals")
        sig_str = i18n.signal_tags(sig_val, L()) if (sig_val and not pd.isna(sig_val)) else "—"
        m4.metric(col_label("signals"), sig_str)

        # Hebrew description
        desc = explain.describe(row)
        if desc:
            st.caption(desc)

        # TradingView Chart
        tv_sym = row.get("tv_symbol")
        if not tv_sym or pd.isna(tv_sym):
            tv_sym = row["ticker"]
        tradingview_chart(str(tv_sym))

        # Add to watchlist bar
        st.divider()
        all_lists = portfolio.load_all()
        watchlists = {name: plist for name, plist in all_lists.items() if plist.kind == portfolio.WATCHLIST}

        wl_options = list(watchlists.keys()) if watchlists else ["מעקב"]

        w_col1, w_col2, w_col3 = st.columns([2, 3, 1.5], vertical_alignment="bottom")

        selected_wl_name = w_col1.selectbox(
            t("Watchlist"),
            options=wl_options,
            key="cf_wl_select",
        )

        trader_note = w_col2.text_input(t("Trader note"), key="cf_note")

        if w_col3.button(t("Add to watchlist"), key="cf_add_wl", use_container_width=True, type="primary"):
            target_name = selected_wl_name or "מעקב"
            if target_name in watchlists:
                plist = watchlists[target_name]
            else:
                plist = portfolio.PList(target_name, portfolio.WATCHLIST, [])

            ticker = str(row["ticker"])
            price = row.get("price")
            price_val = None if (price is None or pd.isna(price)) else float(price)
            note_str = ss.get("cf_note", "").strip() if trader_note is not None else ""

            # Check if ticker already in plist
            found = False
            for r in plist.rows:
                if r.get("ticker") == ticker:
                    found = True
                    r["price"] = price_val
                    if note_str:
                        r["note"] = note_str
                    break

            if not found:
                plist.rows.append({
                    "ticker": ticker,
                    "date": dt.date.today().isoformat(),
                    "price": price_val,
                    "note": note_str,
                })

            portfolio.save(plist)
            st.toast(f"{t('Added to')} {target_name}: {ticker}", icon="✅")

        # Explain Panel
        explain_panel.render(row, selections or {})

    except Exception:
        log.exception("Error rendering fast chart review")
        hebrew_note(t("Could not display fast chart review"))
