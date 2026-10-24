"""Watchlist actions: ticker normalization, adding to watchlists.

Moved verbatim from screener/ui/portfolios_page.py; behavior unchanged."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from screener import portfolio as PF

from screener.ui.common import log, ss, t


NEW_WATCHLIST = "— New watchlist —"


def _by_ticker(df: pd.DataFrame, col: str = "price") -> pd.Series:
    """ticker -> a universe column (prices come from the screener's data: no extra requests)."""
    return df.drop_duplicates("ticker").set_index("ticker")[col]


def add_tickers(plist: PF.PList, text: str, prices: pd.Series) -> tuple[list[str], list[str]]:
    """'aapl, msft nvda' -> (added, not in the screener). Tickers already in the list are skipped."""
    added, unknown = [], []
    for tk in filter(None, (PF.normalize_ticker(x) for x in re.split(r"[,;\s]+", text))):
        if tk not in prices.index:
            unknown.append(tk)
        elif PF.add_to_watchlist(plist, tk, prices[tk]):
            added.append(tk)
    return added, unknown


def _add_and_report(plist: PF.PList, text: str) -> None:
    added, unknown = add_tickers(plist, text, _by_ticker(ss.get("df")))
    try:
        PF.save(plist)
    except Exception as e:
        log.warning("cannot save list %r: %s", plist.name, e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    if added:
        st.toast(f"{t('Added to')} “{plist.name}”: {', '.join(added)}", icon="✅")
    if unknown:
        st.toast(f"{t('Not found in screener')}: {', '.join(unknown)}", icon="⚠️")


def on_add_to_watchlist(ticker: str):
    """From the stock details panel."""
    name = ss.get("add_wl")
    if name == NEW_WATCHLIST:
        name = (ss.get("add_wl_name") or "").strip()
        if not name:
            return
    try:
        lists = PF.load_all()
    except Exception as e:
        log.warning("cannot load lists: %s", e)
        st.toast(t("Cannot save right now, please try again later"), icon="⚠️")
        return
    plist = lists.get(name) or PF.PList(name, PF.WATCHLIST)
    if plist.kind != PF.WATCHLIST:
        st.toast(f"“{name}” {t('is a portfolio, not a watchlist')}", icon="⚠️")
        return
    _add_and_report(plist, ticker)
    ss["add_wl"], ss["add_wl_name"] = name, ""
